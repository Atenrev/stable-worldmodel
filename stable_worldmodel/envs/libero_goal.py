"""LIBERO env for goal-image planning evaluation (the LeWorldModel protocol).

Unlike ``swm/Libero-v0`` (a wrapper around LeRobot's ``LiberoEnv``), this env
drives LIBERO's ``OffScreenRenderEnv`` directly so the evaluation can

* restore a full simulator snapshot (``_set_state``) and a goal snapshot
  (``_set_goal_state``) taken from an expert episode, the way PushT does;
* run past the BDDL task's success without LeRobot's automatic reset;
* score the episode against the goal *state*, not the task predicate.

Snapshots are produced by ``tools/build_libero_goal_eval_set.py`` (nav-jepa):
it replays a dataset episode in the simulator and stores, per step, the
flattened MuJoCo state, the Panda gripper's internal command (it is not part
of the MuJoCo state) and MuJoCo's ``qacc_warmstart``.

Images follow the LeRobot LIBERO dataset convention: both cameras are rotated
180 degrees (``img[::-1, ::-1]``). ``render()`` returns the agent view.
With ``camera_view='both'`` the wrist view is added to the info dict as
``pixels_wrist`` (and the goal's as ``goal_pixels_wrist``), which is what
nav-jepa's two-view world models read.

Success (``terminated``) means the scene matches the goal snapshot:

* end-effector within ``eef_threshold`` metres of the goal end-effector, and
* every object within its threshold of the goal: free objects by xyz
  (``obj_threshold`` m), drawers by slide position (``slide_threshold`` m),
  the stove knob by angle (``hinge_threshold`` rad).

``objects_only_success`` ignores the end-effector.

Two details matter for matching LeRobot LIBERO datasets (checked against
``Atenrev/libero-goal-smolvla-20k``):

* Fixture placement (stove, cabinet, rack) is sampled at ``reset`` from
  ``np.random`` after ``env.seed(seed)`` and lives in the MuJoCo *model*
  (``body_pos`` / ``body_quat``), not in the state. Snapshots therefore carry
  ``body_pos`` / ``body_quat``; with the capture seed the renders match the
  dataset frames before video compression.
* Dataset frames went through H.264 (libx264, crf 30, GOP 2, yuv420p).
  ``obs_codec='h264'`` passes every observation through the same codec
  (``h264_roundtrip``: the last ``codec_window`` frames plus ``codec_pad``
  repeats of the current one as stand-in future frames, then the current
  decoded frame), which brings the WMs' latents of renders much closer to
  those of the dataset frames they were trained on. The info dict also reports
the continuous distances and ``task_success`` (the BDDL predicate).
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ROBOT_PREFIXES = ('robot0_', 'gripper0_', 'mount0_')


def h264_roundtrip(frames, pad: int = 8, crf: int = 30, gop: int = 2) -> np.ndarray:
    """Encode ``frames`` (+ ``pad`` copies of the last one) with libx264 and return the decoded last real frame.

    Mimics LeRobot's dataset video encoding (``h264``, ``crf``, ``g``, ``yuv420p``) for an online stream:
    x264's rate control looks ahead, so the current frame is followed by repeats of itself.
    """
    import io

    import av

    frames = [np.ascontiguousarray(f) for f in frames]
    seq = frames + [frames[-1]] * pad
    h, w = frames[-1].shape[:2]
    buf = io.BytesIO()
    with av.open(buf, 'w', format='mp4') as c:
        s = c.add_stream('h264', 20, options={'g': str(gop), 'crf': str(crf)})
        s.pix_fmt, s.width, s.height = 'yuv420p', w, h
        for f in seq:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format='rgb24')):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    buf.seek(0)
    with av.open(buf) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i == len(frames) - 1:
                return fr.to_ndarray(format='rgb24')
    raise RuntimeError('h264 roundtrip returned too few frames')


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite (x, y, z, w) quaternion -> axis-angle, as in LeRobot's LIBERO processor."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


class LiberoGoal(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        suite: str = 'libero_goal',
        task_id: int = 0,
        camera_view: str = 'agentview',
        resolution: int = 256,
        control_freq: int = 20,
        settle_steps: int = 10,
        eef_threshold: float = 0.04,
        obj_threshold: float = 0.03,
        slide_threshold: float = 0.02,
        hinge_threshold: float = 0.2,
        obs_codec: str | None = 'h264',
        codec_window: int = 8,
        codec_pad: int = 8,
        codec_crf: int = 30,
        render_mode: str = 'rgb_array',
        **kwargs,
    ):
        assert camera_view in ('agentview', 'both'), camera_view
        self.suite_name = suite
        self.camera_view = camera_view
        self.resolution = resolution
        self.control_freq = control_freq
        self.settle_steps = settle_steps
        self.thresholds = dict(
            eef=eef_threshold,
            obj=obj_threshold,
            slide=slide_threshold,
            hinge=hinge_threshold,
        )
        self.render_mode = render_mode
        assert obs_codec in (None, 'none', 'h264'), obs_codec
        self.obs_codec = None if obs_codec == 'none' else obs_codec
        self.codec_window, self.codec_pad, self.codec_crf = codec_window, codec_pad, codec_crf
        self._hist = {'agent': [], 'wrist': []}
        self._cur = None  # (agent, wrist) processed images of the current observation

        self._suite = None
        self._env = None
        self.task_id = None
        self._task_id_default = int(task_id)

        self._goal = None  # goal snapshot summary (dict)
        self._start_dist = None
        self._last_obs = None

        self.action_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {'proprio': spaces.Box(-np.inf, np.inf, (8,), dtype=np.float64)}
        )

    # ------------------------------------------------------------------ setup
    def _benchmark(self):
        if self._suite is None:
            from libero.libero import benchmark

            self._suite = benchmark.get_benchmark_dict()[self.suite_name]()
        return self._suite

    def set_task(self, task_id):
        task_id = int(np.asarray(task_id).reshape(-1)[0])
        if self._env is not None and task_id == self.task_id:
            return
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        if self._env is not None:
            self._env.close()
        task = self._benchmark().get_task(task_id)
        bddl = os.path.join(
            get_libero_path('bddl_files'), task.problem_folder, task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_names=['agentview', 'robot0_eye_in_hand'],
            camera_heights=self.resolution,
            camera_widths=self.resolution,
            control_freq=self.control_freq,
            hard_reset=False,
            ignore_done=True,
        )
        self.task_id = task_id
        self.task_description = task.language
        self._init_states = self._benchmark().get_task_init_states(task_id)
        self._joints = self._object_joints()
        self._goal = None
        self._start_dist = None

    def _object_joints(self):
        """(name, kind, qpos_slice) for every non-robot joint (objects, drawers, knobs)."""
        import mujoco

        model = self.sim.model
        out = []
        for j in range(model.njnt):
            name = model.joint_id2name(j)
            if name is None or name.startswith(ROBOT_PREFIXES):
                continue
            adr = int(model.jnt_qposadr[j])
            jt = int(model.jnt_type[j])
            if jt == int(mujoco.mjtJoint.mjJNT_FREE):
                out.append((name, 'obj', slice(adr, adr + 3)))
            elif jt == int(mujoco.mjtJoint.mjJNT_SLIDE):
                out.append((name, 'slide', slice(adr, adr + 1)))
            elif jt == int(mujoco.mjtJoint.mjJNT_HINGE):
                out.append((name, 'hinge', slice(adr, adr + 1)))
        return out

    @property
    def sim(self):
        return self._env.sim

    @property
    def _gripper(self):
        return self._env.robots[0].gripper

    # -------------------------------------------------------------- snapshots
    def get_snapshot(self) -> dict:
        """Everything needed to restore the simulator exactly (used by the eval-set builder)."""
        return {
            'sim_state': self.sim.get_state().flatten().copy(),
            'gripper_action': np.asarray(self._gripper.current_action, dtype=np.float64).copy(),
            'qacc_warmstart': self.sim.data.qacc_warmstart.copy(),
            'body_pos': self.sim.model.body_pos.flatten().copy(),
            'body_quat': self.sim.model.body_quat.flatten().copy(),
        }

    def _restore(self, sim_state, gripper_action=None, qacc_warmstart=None, body_pos=None, body_quat=None):
        model = self.sim.model
        if body_pos is not None:  # fixture placement lives in the model, not the state
            model.body_pos[:] = np.asarray(body_pos, dtype=np.float64).reshape(-1)[: model.nbody * 3].reshape(-1, 3)
        if body_quat is not None:
            model.body_quat[:] = np.asarray(body_quat, dtype=np.float64).reshape(-1)[: model.nbody * 4].reshape(-1, 4)
        sim_state = np.asarray(sim_state, dtype=np.float64).reshape(-1)
        n = len(self.sim.get_state().flatten())
        sim_state = sim_state[:n]
        obs = self._env.regenerate_obs_from_state(sim_state)
        if gripper_action is not None:
            self._gripper.current_action = np.asarray(gripper_action, dtype=np.float64).reshape(-1).copy()
        if qacc_warmstart is not None:
            self.sim.data.qacc_warmstart[:] = np.asarray(qacc_warmstart, dtype=np.float64).reshape(-1)[: self.sim.model.nv]
        self._env.env.timestep = 0
        self._env.env.done = False
        self._set_obs(obs, reset_history=True)
        return obs

    def _set_obs(self, obs, reset_history=False):
        """Make ``obs`` current: rotate the cameras 180deg and (optionally) pass them through the video codec."""
        self._last_obs = obs
        raw = {
            'agent': np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]),
            'wrist': np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]),
        }
        out = {}
        for view, img in raw.items():
            if reset_history:
                self._hist[view] = []
            self._hist[view] = (self._hist[view] + [img])[-self.codec_window:]
            if self.obs_codec == 'h264' and (view == 'agent' or self.camera_view == 'both'):
                out[view] = h264_roundtrip(self._hist[view], pad=self.codec_pad, crf=self.codec_crf)
            else:
                out[view] = img
        self._cur = (out['agent'], out['wrist'])

    def _scene(self, obs) -> dict:
        qpos = self.sim.data.qpos
        return {
            'eef': np.asarray(obs['robot0_eef_pos'], dtype=np.float64).copy(),
            'joints': {name: qpos[s].copy() for name, _, s in self._joints},
        }

    def _images(self, obs=None):
        """Processed (rotated, optionally codec-roundtripped) agent and wrist images of the current observation."""
        return self._cur

    def _set_goal_state(self, goal_state, goal_gripper_action=None, goal_qacc_warmstart=None,
                        goal_body_pos=None, goal_body_quat=None):
        """Load the goal snapshot, record its scene and images, then restore the current state."""
        current = self.get_snapshot() if self._env is not None else None
        obs = self._restore(goal_state, goal_gripper_action, goal_qacc_warmstart, goal_body_pos, goal_body_quat)
        agent, wrist = self._images(obs)
        self._goal = {
            **self._scene(obs),
            'pixels': agent,
            'pixels_wrist': wrist,
            'task_success': bool(self._env.check_success()),
        }
        if current is not None:
            self._restore(**current)

    def _set_state(self, state, gripper_action=None, qacc_warmstart=None, body_pos=None, body_quat=None):
        self._restore(state, gripper_action, qacc_warmstart, body_pos, body_quat)
        # distances of the start state to the goal = the 'do nothing' baseline
        m = self.eval_state()
        self._start_dist = {k: m[k] for k in ('eef_dist', 'obj_dist', 'slide_dist', 'hinge_dist')} if m else None

    # --------------------------------------------------------------- metrics
    def eval_state(self) -> dict:
        if self._goal is None or self._last_obs is None:
            return {}
        cur = self._scene(self._last_obs)
        th = self.thresholds
        eef_dist = float(np.linalg.norm(cur['eef'] - self._goal['eef']))
        obj_dist, obj_ok = 0.0, True
        art_ok = True
        max_slide, max_hinge = 0.0, 0.0
        for name, kind, _ in self._joints:
            d = float(np.linalg.norm(cur['joints'][name] - self._goal['joints'][name]))
            if kind == 'obj':
                obj_dist = max(obj_dist, d)
                obj_ok &= d < th['obj']
            elif kind == 'slide':
                max_slide = max(max_slide, d)
                art_ok &= d < th['slide']
            else:
                max_hinge = max(max_hinge, d)
                art_ok &= d < th['hinge']
        objects_only = bool(obj_ok and art_ok)
        start = getattr(self, '_start_dist', None) or {}
        return {
            'start_eef_dist': start.get('eef_dist', np.nan),
            'start_obj_dist': start.get('obj_dist', np.nan),
            'eef_dist': eef_dist,
            'obj_dist': obj_dist,
            'slide_dist': max_slide,
            'hinge_dist': max_hinge,
            'objects_only_success': objects_only,
            'state_success': bool(objects_only and eef_dist < th['eef']),
            'task_success': bool(self._env.check_success()),
            'goal_task_success': bool(self._goal['task_success']),
        }

    # -------------------------------------------------------------- gym API
    def _obs_info(self, obs):
        pos = np.asarray(obs['robot0_eef_pos'], dtype=np.float64)
        proprio = np.concatenate(
            [pos, _quat2axisangle(obs['robot0_eef_quat']), np.asarray(obs['robot0_gripper_qpos'], dtype=np.float64)]
        )
        info = {}
        if self.camera_view == 'both':
            _, wrist = self._images(obs)
            info['pixels_wrist'] = wrist
            if self._goal is not None:
                info['goal_pixels_wrist'] = self._goal['pixels_wrist']
            else:
                info['goal_pixels_wrist'] = np.zeros_like(wrist)
        info['goal'] = (
            self._goal['pixels'] if self._goal is not None
            else np.zeros((self.resolution, self.resolution, 3), np.uint8)
        )
        metrics = self.eval_state()
        for k in ('eef_dist', 'obj_dist', 'slide_dist', 'hinge_dist', 'start_eef_dist', 'start_obj_dist'):
            info[k] = np.float32(metrics.get(k, np.nan))
        for k in ('state_success', 'objects_only_success', 'task_success', 'goal_task_success'):
            info[k] = bool(metrics.get(k, False))
        return {'proprio': proprio}, info, metrics

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self._env is None or 'task_id' in options:
            self.set_task(options.get('task_id', self._task_id_default if self._env is None else self.task_id))
        self._env.seed(seed if seed is not None else 0)
        self._env.reset()
        init_id = int(options.get('init_state_id', 0)) % len(self._init_states)
        obs = self._env.set_init_state(self._init_states[init_id])
        noop = np.zeros(7, dtype=np.float32)
        noop[-1] = -1.0
        for _ in range(self.settle_steps):
            obs, _, _, _ = self._env.step(noop)
        self._env.env.timestep = 0
        self._goal = None
        self._set_obs(obs, reset_history=True)
        out, info, _ = self._obs_info(obs)
        return out, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        obs, _, _, _ = self._env.step(action)
        self._set_obs(obs)
        out, info, metrics = self._obs_info(obs)
        terminated = bool(metrics.get('state_success', False))
        reward = -float(metrics.get('eef_dist', 0.0) + metrics.get('obj_dist', 0.0))
        return out, reward, terminated, False, info

    def render(self):
        if self._cur is None:
            return np.zeros((self.resolution, self.resolution, 3), np.uint8)
        return self._cur[0]

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
