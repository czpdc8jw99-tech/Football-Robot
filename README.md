# Football Robot (Web)

Browser-based MuJoCo (WASM) multi-robot simulator driven by an ONNX policy, with
global + per-robot motion controls and camera follow.

## 🌐 Demo

| |
|---|
| GitHub Pages: `https://czpdc8jw99-tech.github.io/Football-Robot/` |

## ✨ Features

- **MuJoCo in the browser (WASM)** + Three.js rendering
- **ONNX Runtime Web (WASM)** policy inference
- **Multi-robot scene generation** (draft → generate/apply)
- **Per-robot controls**
  - Policy selection (per robot)
  - Motion selection (per robot)
  - Position (X/Y) before generating
- **Global motion controller**
  - Select one motion to apply to **all generated robots**
  - Switching motion: **force return to `default` → auto-play target motion**
  - Shows a global `Pending: ...` hint while waiting
- **Restart motion**
  - Clicking the currently-selected motion can restart it (return to `default` then replay)
- **Camera**
  - Focus & Follow the selected robot (WASDQE cancels follow)
- **Custom motions upload**
  - Upload motion JSON clips from the UI

## 🚀 Quick start

```bash
npm install
npm run dev
```

Build:

```bash
npm run build
```

## 🧭 Project structure

- `src/views/Demo.vue`: UI (policies, motions, multi-robot controls)
- `src/simulation/main.js`: MuJoCo + rendering + main loop
- `src/simulation/mujocoUtils.js`: scene/policy loading utilities
- `src/simulation/policyRunner.js`: observation pipeline + action target output
- `src/simulation/onnxHelper.js`: ONNX Runtime Web session creation + inference
- `public/examples/scenes/`: MJCF files + meshes staged into MuJoCo MEMFS
- `public/examples/checkpoints/`: policy config JSON, ONNX file, motion clips
- `scripts/`: helper scripts for managing motion clips/assets
- `.github/workflows/deploy.yml`: GitHub Pages build + deploy workflow

## 🛠 Debugging

- **Enable extra debug logs**: in browser devtools console run `window.__FOOTBALL_ROBOT_DEBUG__ = true`

## 🧩 Add your own robot / policy / motions (high level)

1. **Scene**
   - Put MJCF + assets under `public/examples/scenes/<robot>/`
   - Add all files into `public/examples/scenes/files.json` so they get preloaded
2. **Policy**
   - Put policy config JSON + ONNX under `public/examples/checkpoints/<robot>/`
   - Ensure `policy_joint_names`, `obs_config`, PD gains, and `default_joint_pos` match your model
3. **Motions (optional)**
   - Provide `tracking.motions_path` index JSON and per-clip files under `motions/`

## 🤝 Contribution

| |
|---|
| We welcome external contributions! If you have suggestions, wish to add support for a new asset configuration, or improve alignment parameters, please feel free to open an issue or submit a pull request. |
