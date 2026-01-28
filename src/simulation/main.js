import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { downloadExampleScenesFolder, getPosition, getQuaternion, toMujocoPos, reloadScene, reloadPolicy, reloadPolicyForRobot } from './mujocoUtils.js';
import { generateMultiRobotXML } from './multiRobotGenerator.js';

const defaultPolicy = "./examples/checkpoints/g1/tracking_policy_amass.json";

function isDebugEnabled() {
  return typeof window !== 'undefined' && window.__FOOTBALL_ROBOT_DEBUG__ === true;
}

function debugLog(...args) {
  if (isDebugEnabled()) console.log(...args);
}

function debugWarn(...args) {
  if (isDebugEnabled()) console.warn(...args);
}

function debugError(...args) {
  if (isDebugEnabled()) console.error(...args);
}

export class MuJoCoDemo {
  constructor(mujoco) {
    this.mujoco = mujoco;
    mujoco.FS.mkdir('/working');
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');

    this.params = {
      paused: true,
      current_motion: 'default'
    };
    this.policyRunner = null; // Backward compatibility (single-robot mode)
    this.policyRunners = []; // Multi-robot mode
    this.kpPolicy = null;
    this.kdPolicy = null;
    this.robotPolicyParams = []; // Per-robot PD params / policy metadata
    this.actionTarget = null;
    this.model = null;
    this.data = null;
    this.simulation = null;
    this.currentPolicyPath = defaultPolicy;

    this.bodies = {};
    this.lights = {};

    this.container = document.getElementById('mujoco-container');

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 100);
    this.camera.name = 'PerspectiveCamera';
    // Football field view: focus on the robot and show more of the field.
    // Camera pose: look at the robot from the upper-right-front direction (keep robot centered).
    // Three.js coordinates: +X right, +Y up, +Z forward.
    // Robot position: Three.js (0, 0.8, 0).
    // Camera position: look from (4, 3, 5) toward (0, 0.8, 0).
    this.camera.position.set(4.0, 3.0, 5.0);
    this.scene.add(this.camera);

    // Football field background (sky blue).
    this.scene.background = new THREE.Color(0.5, 0.7, 1.0);
    this.scene.fog = null;

    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.1);
    this.ambientLight.name = 'AmbientLight';
    this.scene.add(this.ambientLight);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderScale = 2.0;
    this.renderer.setPixelRatio(this.renderScale);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.simStepHz = 0;
    this._stepFrameCount = 0;
    this._stepLastTime = performance.now();
    this._lastRenderTime = 0;

    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    // Focus on the robot (MuJoCo: (0, 0, 0.8) -> Three.js: (0, 0.8, 0)).
    // getPosition conversion: MuJoCo(x, y, z) -> Three.js(x, z, -y).
    // NOTE: target must be set to the robot position (not the goal).
    this.controls.target.set(0, 0.8, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    // Force-update to apply the target immediately.
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));

    // WASDQE keyboard camera controls.
    this.keys = {};
    this.cameraSpeed = 0.5; // movement speed
    
    // Create handlers bound to this instance.
    this._handleKeyDown = (e) => {
      // Avoid triggering while typing in input fields.
      const target = e.target || e.srcElement;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return;
      }
      
      const key = e.key.toLowerCase();
      if (['w', 'a', 's', 'd', 'q', 'e'].includes(key)) {
        e.preventDefault();
        e.stopPropagation();
        // If follow mode is enabled, any manual control key cancels follow immediately.
        if (this.followEnabled) {
          this.setFollowEnabled(false);
        }
        this.keys[key] = true;
      }
    };
    
    this._handleKeyUp = (e) => {
      const key = e.key.toLowerCase();
      if (['w', 'a', 's', 'd', 'q', 'e'].includes(key)) {
        this.keys[key] = false;
      }
    };
    
    // Listen on the container so the canvas can receive keyboard events.
    if (this.container) {
      this.container.setAttribute('tabindex', '0');
      this.container.style.outline = 'none';
      this.container.addEventListener('keydown', this._handleKeyDown);
      this.container.addEventListener('keyup', this._handleKeyUp);
      // Focus on click so the canvas can receive keyboard events.
      this.container.addEventListener('click', () => {
        this.container.focus();
        debugLog('Canvas focused; WASDQE enabled.');
      });
      // Also focus when the mouse enters.
      this.container.addEventListener('mouseenter', () => {
        this.container.focus();
      });
    }
    
    // Also listen on document as a fallback (capture mode).
    document.addEventListener('keydown', this._handleKeyDown, true);
    document.addEventListener('keyup', this._handleKeyUp, true);

    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);

    // Track whether the mouse is dragging (to decide if OrbitControls should control camera).
    this._isMouseDragging = false;
    this._mouseDownHandler = () => {
      this._isMouseDragging = true;
    };
    this._mouseUpHandler = () => {
      this._isMouseDragging = false;
    };
    // Track whether the mouse wheel is active (to decide if OrbitControls should control zoom).
    this._isWheeling = false;
    this._wheelTimeout = null;
    this._wheelHandler = () => {
      this._isWheeling = true;
      // Clear previous timeout.
      if (this._wheelTimeout) {
        clearTimeout(this._wheelTimeout);
      }
      // Reset wheel state after 200ms.
      this._wheelTimeout = setTimeout(() => {
        this._isWheeling = false;
      }, 200);
    };
    // Mouse event listeners.
    if (this.renderer.domElement) {
      this.renderer.domElement.addEventListener('mousedown', this._mouseDownHandler);
      this.renderer.domElement.addEventListener('mouseup', this._mouseUpHandler);
      this.renderer.domElement.addEventListener('mouseleave', this._mouseUpHandler); // stop dragging when leaving canvas
      this.renderer.domElement.addEventListener('wheel', this._wheelHandler, { passive: true }); // wheel zoom tracking
    }

    this.followEnabled = false;
    this.followHeight = 0.75;
    this.followLerp = 0.05;
    this.followTarget = new THREE.Vector3();
    this.followTargetDesired = new THREE.Vector3();
    this.followDelta = new THREE.Vector3();
    this.followOffset = new THREE.Vector3();
    this.followInitialized = false;
    this.followBodyId = null;
    this.followDistance = this.camera.position.distanceTo(this.controls.target);

    this.lastSimState = {
      bodies: new Map(),
      lights: new Map(),
      tendons: {
        numWraps: 0,
        matrix: new THREE.Matrix4()
      }
    };

    this.renderer.setAnimationLoop(this.render.bind(this));

    this.reloadScene = reloadScene.bind(this);
    this.reloadPolicy = reloadPolicy.bind(this);
    this.reloadPolicyForRobot = reloadPolicyForRobot.bind(this);
    
    // Multi-robot configuration.
    this.robotConfigs = []; // per-robot initial poses: {x, y, z}
    this.robotPelvisBodyIds = []; // pelvis body IDs (for fast focus/follow)
    this.followRobotIndex = 0; // focused robot index (0-based)
    // Multi-robot joint mappings.
    this.robotJointMappings = []; // per-robot joint mapping entries
  }
  
  /**
   * Generate a multi-robot scene.
   * Builds a MuJoCo XML scene by cloning the base robot according to robotConfigs.
   * @param {Array<{x: number, y: number, z: number}>} robotConfigs - per-robot initial positions
   * @throws {Error} if configs are empty or robot count is too large
   */
  async generateMultiRobotScene(robotConfigs) {
    if (!robotConfigs || robotConfigs.length === 0) {
      throw new Error('Robot configs cannot be empty');
    }
    
    if (robotConfigs.length > 11) {
      throw new Error('Maximum 11 robots allowed');
    }
    
    this.robotConfigs = robotConfigs;
    
    // Generate multi-robot XML.
    const baseXmlPath = './examples/scenes/g1/g1.xml';
    const generatedXml = await generateMultiRobotXML(baseXmlPath, robotConfigs);
    
    // Write generated XML into the MuJoCo filesystem.
    const targetPath = '/working/g1/g1.xml';
    this.mujoco.FS.writeFile(targetPath, generatedXml);
    
    debugLog(`Generated multi-robot scene with ${robotConfigs.length} robots`);
  }
  
  /**
   * Set initial positions for multiple robots.
   * Call after scene load: sets each robot freejoint qpos (position + quaternion),
   * and records each robot's pelvis body ID for focus/follow.
   */
  setMultiRobotInitialPositions() {
    if (!this.robotConfigs || this.robotConfigs.length <= 1) {
      return; // single robot: no-op
    }
    
    if (!this.model || !this.data || !this.simulation) {
      console.warn('Cannot set robot positions: model/data not loaded');
      return;
    }
    
    const textDecoder = new TextDecoder();
    const namesArray = new Uint8Array(this.model.names);
    const qpos = this.simulation.qpos;
    
    // Set each robot initial position and record pelvis body IDs.
    this.robotPelvisBodyIds = []; // reset
    this.robotConfigs.forEach((config, index) => {
      const robotPrefix = index === 0 ? 'pelvis' : `robot${index + 1}_pelvis`;
      
      // Find the pelvis body ID.
      let pelvisBodyId = -1;
      for (let b = 0; b < this.model.nbody; b++) {
        let start_idx = this.model.name_bodyadr[b];
        let end_idx = start_idx;
        while (end_idx < namesArray.length && namesArray[end_idx] !== 0) {
          end_idx++;
        }
        let name_buffer = namesArray.subarray(start_idx, end_idx);
        const bodyName = textDecoder.decode(name_buffer);
        
        if (bodyName === robotPrefix) {
          pelvisBodyId = b;
          break;
        }
      }
      
      // Store pelvis body ID.
      if (pelvisBodyId >= 0) {
        this.robotPelvisBodyIds[index] = pelvisBodyId;
      }
      
      if (pelvisBodyId < 0) {
        console.warn(`Could not find body "${robotPrefix}" for robot ${index + 1}`);
        return;
      }
      
      // Find the corresponding freejoint (use the first joint under the pelvis body).
      // The freejoint body ID should match the pelvis body ID.
      for (let j = 0; j < this.model.njnt; j++) {
        if (this.model.jnt_bodyid[j] === pelvisBodyId) {
          const qposAdr = this.model.jnt_qposadr[j];
          if (qposAdr >= 0 && qposAdr + 6 < qpos.length) {
            // Position (x, y, z)
            qpos[qposAdr + 0] = config.x;
            qpos[qposAdr + 1] = config.y;
            qpos[qposAdr + 2] = config.z;
            // Quaternion (w, x, y, z) - default upright orientation
            qpos[qposAdr + 3] = 1.0; // w
            qpos[qposAdr + 4] = 0.0; // x
            qpos[qposAdr + 5] = 0.0; // y
            qpos[qposAdr + 6] = 0.0; // z
            
            debugLog(`Set robot ${index + 1} (${robotPrefix}) initial position: (${config.x}, ${config.y}, ${config.z})`);
            break;
          }
        }
      }
    });
    
    // Update physics state.
    this.simulation.forward();
  }

  async init() {
    await downloadExampleScenesFolder(this.mujoco);
    await this.reloadScene('g1/g1.xml');
    this.updateFollowBodyId();
    await this.reloadPolicy(defaultPolicy);
    this.alive = true;
  }

  async reload(mjcf_path) {
    await this.reloadScene(mjcf_path);
    this.updateFollowBodyId();
    this.timestep = this.model.opt.timestep;
    this.decimation = Math.max(1, Math.round(0.02 / this.timestep));

    debugLog('timestep:', this.timestep, 'decimation:', this.decimation);

    await this.reloadPolicy(this.currentPolicyPath ?? defaultPolicy);
    this.alive = true;
  }

  setFollowEnabled(enabled) {
    this.followEnabled = Boolean(enabled);
    this.followInitialized = false;
    if (this.followEnabled) {
      // Follow enabled: mouse controls everything (rotate/pan/zoom); keyboard does nothing.
      this.controls.enableRotate = true;
      this.controls.enablePan = true;
      this.controls.enableZoom = true; // mouse wheel zoom
      
      // Compute camera offset relative to target.
      this.followOffset.subVectors(this.camera.position, this.controls.target);
      if (this.followOffset.lengthSq() === 0) {
        this.followOffset.set(0, 0, 1);
      }
      this.followOffset.setLength(this.followDistance);
      // Persist the user-adjusted offset (after mouse movement).
      this._userFollowOffset = this.followOffset.clone();
      this.camera.position.copy(this.controls.target).add(this.followOffset);
      this.controls.update();
    } else {
      // Follow disabled: mouse + keyboard can both control the view.
      // Mouse: rotate/zoom; Keyboard: WASDQE translates target + camera.
      this.controls.enableRotate = true; // mouse drag rotate
      this.controls.enablePan = false; // disable mouse pan (avoid conflicts)
      this.controls.enableZoom = true; // mouse wheel zoom
    }
  }

  /**
   * Update the follow body ID.
   * Uses the currently selected robot index to resolve the corresponding pelvis body ID.
   * Prefers multi-robot mappings when available, otherwise falls back to single-robot mode.
   */
  updateFollowBodyId() {
    // Multi-robot: prefer the configured robot index.
    if (this.robotPelvisBodyIds && this.robotPelvisBodyIds.length > 0) {
      const robotIndex = this.followRobotIndex || 0;
      if (robotIndex >= 0 && robotIndex < this.robotPelvisBodyIds.length) {
        const bodyId = this.robotPelvisBodyIds[robotIndex];
        if (Number.isInteger(bodyId) && bodyId >= 0) {
          this.followBodyId = bodyId;
          return;
        }
      }
    }
    
    // Method 1: use pelvis_body_id.
    if (Number.isInteger(this.pelvis_body_id)) {
      this.followBodyId = this.pelvis_body_id;
      return;
    }
    
    // Method 2: search by body name.
    if (this.bodies) {
      for (const bodyId in this.bodies) {
        const body = this.bodies[bodyId];
        if (body && body.name === 'pelvis') {
          this.followBodyId = parseInt(bodyId);
          return;
        }
      }
    }
    
    // Method 3: scan for a pelvis-like body name (no position constraints).
    if (this.model && this.lastSimState && this.bodies) {
      for (let testId = 1; testId < this.model.nbody; testId++) {
        const body = this.bodies[testId];
        if (body && body.name && body.name.includes('pelvis')) {
          this.followBodyId = testId;
          return;
        }
      }
    }
    
    // Method 4: last resort - use body ID 1 (may not be pelvis).
    if (this.model && this.model.nbody > 1) {
      this.followBodyId = 1;
    }
  }
  
  // Focus camera on the selected robot.
  focusOnRobot(robotIndex = 0) {
    // Set which robot is selected for follow/focus.
    if (this.robotPelvisBodyIds && this.robotPelvisBodyIds.length > 0) {
      const index = Math.max(0, Math.min(robotIndex, this.robotPelvisBodyIds.length - 1));
      this.followRobotIndex = index;
      
      // Update follow body ID.
      this.updateFollowBodyId();
      
      // If follow is enabled, update camera immediately.
      if (this.followEnabled) {
        this.followInitialized = false;
        this.updateCameraFollow();
      } else {
        // Follow disabled: move camera to a good pose relative to the robot.
        const bodyId = this.robotPelvisBodyIds[index];
        if (Number.isInteger(bodyId) && bodyId >= 0) {
          const cached = this.lastSimState?.bodies?.get(bodyId);
          if (cached && cached.position) {
            // Set camera target to the pelvis position (slightly raised).
            const robotPos = cached.position;
            this.controls.target.set(robotPos.x, robotPos.y + 0.8, robotPos.z);
            
            // Use an offset similar to the initial camera pose.
            const offset = new THREE.Vector3(4.0, 2.2, 5.0);
            this.camera.position.copy(this.controls.target).add(offset);
            this.controls.update();
          }
        }
      }
      
      debugLog(`Focused on robot ${index + 1} (body ID: ${this.robotPelvisBodyIds[index]})`);
    } else {
      // Single-robot mode: use a reasonable distance/angle.
      if (Number.isInteger(this.pelvis_body_id)) {
        const cached = this.lastSimState?.bodies?.get(this.pelvis_body_id);
        if (cached && cached.position) {
          const robotPos = cached.position;
          this.controls.target.set(robotPos.x, robotPos.y + 0.8, robotPos.z);
          
          // Use an offset similar to the initial camera pose.
          const offset = new THREE.Vector3(4.0, 2.2, 5.0);
          this.camera.position.copy(this.controls.target).add(offset);
          this.controls.update();
        }
      }
    }
  }
  
  /**
   * Set the robot index to follow.
   * When camera follow is enabled, selects which robot to follow.
   * @param {number} robotIndex - robot index (0-based)
   */
  setFollowRobotIndex(robotIndex) {
    this.followRobotIndex = Math.max(0, robotIndex);
    this.updateFollowBodyId();
    if (this.followEnabled) {
      this.followInitialized = false;
    }
  }

  updateCameraFollow() {
    if (!this.followEnabled) {
      return;
    }
    const bodyId = Number.isInteger(this.followBodyId) ? this.followBodyId : null;
    if (bodyId === null) {
      return;
    }
    const cached = this.lastSimState.bodies.get(bodyId);
    if (!cached) {
      return;
    }
    this.followTargetDesired.set(cached.position.x, this.followHeight, cached.position.z);
    if (!this.followInitialized) {
      this.followTarget.copy(this.followTargetDesired);
      this.followInitialized = true;
    } else {
      this.followTarget.lerp(this.followTargetDesired, this.followLerp);
    }

    this.followDelta.subVectors(this.followTarget, this.controls.target);
    this.controls.target.copy(this.followTarget);
    this.camera.position.add(this.followDelta);
  }

  async main_loop() {
    // Do not rely on an "initial mode" flag; detect mode dynamically in the loop.
    if (!this.policyRunner && (!this.policyRunners || this.policyRunners.length === 0)) {
      return;
    }

    while (this.alive) {
      const loopStart = performance.now();
      
      // Detect multi-robot mode dynamically (robotJointMappings may be populated after init).
      const isMultiRobot = this.robotJointMappings && this.robotJointMappings.length > 1;
      const hasPolicyRunner = isMultiRobot 
        ? (this.policyRunners && this.policyRunners.length > 0)
        : this.policyRunner;
      
      // Only log the first time we detect a mode (avoid spamming).
      if (!this._lastMultiRobotMode && isMultiRobot) {
        debugLog('Multi-robot mode detected:', {
          robotJointMappingsLength: this.robotJointMappings?.length,
          policyRunnersLength: this.policyRunners?.length,
          isMultiRobot: true
        });
        this._lastMultiRobotMode = true;
      } else if (this._lastMultiRobotMode === undefined && !isMultiRobot) {
        // Only log once for single-robot mode (avoid false positives during init).
        if (this.robotJointMappings && this.robotJointMappings.length === 0) {
          // Init stage: do not log.
        } else {
          debugLog('Single-robot mode:', {
            robotJointMappingsLength: this.robotJointMappings?.length,
            hasPolicyRunner: !!this.policyRunner,
            isMultiRobot: false
          });
          this._lastMultiRobotMode = false;
        }
      }

      if (!this.params.paused && this.model && this.data && this.simulation && hasPolicyRunner) {
        // State read + inference (one PolicyRunner per robot).
        let actionTargets = [];
        if (isMultiRobot) {
          // Multi-robot: run inference per robot.
          try {
            for (let robotIdx = 0; robotIdx < this.robotJointMappings.length; robotIdx++) {
              if (!this.policyRunners[robotIdx]) {
                console.warn(`Policy runner not found for robot ${robotIdx + 1}`);
                continue;
              }
              const state = this.readPolicyStateForRobot(robotIdx);
              if (!state) {
                console.warn(`Failed to read state for robot ${robotIdx + 1}`);
                continue;
              }
              const actionTarget = await this.policyRunners[robotIdx].step(state);
              // Validate the actionTarget (Array or Float32Array).
              if (!actionTarget || (!Array.isArray(actionTarget) && !(actionTarget instanceof Float32Array)) || actionTarget.length === 0) {
                console.error(`Policy runner ${robotIdx + 1} returned invalid actionTarget:`, {
                  actionTarget,
                  type: typeof actionTarget,
                  isArray: Array.isArray(actionTarget),
                  isFloat32Array: actionTarget instanceof Float32Array,
                  length: actionTarget?.length
                });
                continue;
              }
              actionTargets[robotIdx] = actionTarget;
            }
            // Validate actionTargets array.
            if (actionTargets.length !== this.robotJointMappings.length) {
              debugWarn('actionTargets length mismatch:', {
                actionTargetsLength: actionTargets.length,
                mappingsLength: this.robotJointMappings.length,
                actionTargetsKeys: Object.keys(actionTargets)
              });
            }
            // Backward compatibility: keep robot 0 actionTarget in this.actionTarget.
            this.actionTarget = actionTargets[0];
          } catch (e) {
            console.error('Inference error in main loop:', e);
            this.alive = false;
            break;
          }
        } else {
          // Single robot: original code path.
          const state = this.readPolicyState();
          try {
            this.actionTarget = await this.policyRunner.step(state);
            actionTargets = [this.actionTarget]; // keep array shape consistent
          } catch (e) {
            console.error('Inference error in main loop:', e);
            this.alive = false;
            break;
          }
        }

        for (let substep = 0; substep < this.decimation; substep++) {
          if (this.control_type === 'joint_position') {
            if (isMultiRobot) {
              // Multi-robot: apply control to all robots.
              for (let robotIdx = 0; robotIdx < this.robotJointMappings.length; robotIdx++) {
                const mapping = this.robotJointMappings[robotIdx];
                if (!mapping) {
                  if (substep === 0 && robotIdx > 0) {
                    debugWarn(`Mapping not found for robot ${robotIdx + 1}`);
                  }
                  continue;
                }
                
                const actionTarget = actionTargets[robotIdx];
                
                if (!actionTarget) {
                  // Missing actionTarget: log details for debugging.
                  if (substep === 0 && robotIdx > 0) {
                    debugError(`ActionTarget not found for robot ${robotIdx + 1}:`, {
                      actionTargetsLength: actionTargets.length,
                      actionTargetsKeys: Object.keys(actionTargets),
                      actionTargetsHasIndex: robotIdx in actionTargets,
                      robotIdx
                    });
                  }
                  continue;
                }
                
                // Validate the actionTarget (Array or Float32Array).
                if (!Array.isArray(actionTarget) && !(actionTarget instanceof Float32Array)) {
                  if (substep === 0 && robotIdx > 0) {
                    debugError(`ActionTarget for robot ${robotIdx + 1} is not an array:`, {
                      type: typeof actionTarget,
                      constructor: actionTarget?.constructor?.name,
                      value: actionTarget
                    });
                  }
                  continue;
                }
                
                if (actionTarget.length !== mapping.numActions) {
                  if (substep === 0 && robotIdx > 0) {
                    debugError(`ActionTarget length mismatch for robot ${robotIdx + 1}:`, {
                      actionTargetLength: actionTarget.length,
                      numActions: mapping.numActions
                    });
                  }
                  continue;
                }
                
                for (let i = 0; i < mapping.numActions; i++) {
                  const qposAdr = mapping.qpos_adr_policy[i];
                  const qvelAdr = mapping.qvel_adr_policy[i];
                  const ctrlAdr = mapping.ctrl_adr_policy[i];
                  
                  // Ensure actionTarget[i] is a valid number.
                  const targetJpos = (actionTarget[i] !== undefined && actionTarget[i] !== null) ? actionTarget[i] : 0.0;
                  // Allow per-robot PD parameters (for independent policies).
                  const robotParams = this.robotPolicyParams?.[robotIdx] ?? null;
                  const kpArr = robotParams?.kp ?? this.kpPolicy;
                  const kdArr = robotParams?.kd ?? this.kdPolicy;
                  const kp = kpArr ? kpArr[i] : 0.0;
                  const kd = kdArr ? kdArr[i] : 0.0;
                  const torque = kp * (targetJpos - this.simulation.qpos[qposAdr]) 
                               + kd * (0 - this.simulation.qvel[qvelAdr]);
                  
                  let ctrlValue = torque;
                  const ctrlRange = this.model?.actuator_ctrlrange;
                  if (ctrlRange && ctrlRange.length >= (ctrlAdr + 1) * 2) {
                    const min = ctrlRange[ctrlAdr * 2];
                    const max = ctrlRange[(ctrlAdr * 2) + 1];
                    if (Number.isFinite(min) && Number.isFinite(max) && min < max) {
                      ctrlValue = Math.min(Math.max(ctrlValue, min), max);
                    }
                  }
                  this.simulation.ctrl[ctrlAdr] = ctrlValue;
                }
              }
            } else {
              // Single-robot (original logic).
              for (let i = 0; i < this.numActions; i++) {
                const qpos_adr = this.qpos_adr_policy[i];
                const qvel_adr = this.qvel_adr_policy[i];
                const ctrl_adr = this.ctrl_adr_policy[i];

                const targetJpos = this.actionTarget ? this.actionTarget[i] : 0.0;
                const kp = this.kpPolicy ? this.kpPolicy[i] : 0.0;
                const kd = this.kdPolicy ? this.kdPolicy[i] : 0.0;
                const torque = kp * (targetJpos - this.simulation.qpos[qpos_adr]) + kd * (0 - this.simulation.qvel[qvel_adr]);
                let ctrlValue = torque;
                const ctrlRange = this.model?.actuator_ctrlrange;
                if (ctrlRange && ctrlRange.length >= (ctrl_adr + 1) * 2) {
                  const min = ctrlRange[ctrl_adr * 2];
                  const max = ctrlRange[(ctrl_adr * 2) + 1];
                  if (Number.isFinite(min) && Number.isFinite(max) && min < max) {
                    ctrlValue = Math.min(Math.max(ctrlValue, min), max);
                  }
                }
                this.simulation.ctrl[ctrl_adr] = ctrlValue;
              }
            }
          } else if (this.control_type === 'torque') {
            console.error('Torque control not implemented yet.');
          }

          const applied = this.simulation.qfrc_applied;
          for (let i = 0; i < applied.length; i++) {
            applied[i] = 0.0;
          }

          const dragged = this.dragStateManager.physicsObject;
          if (dragged && dragged.bodyID) {
            for (let b = 0; b < this.model.nbody; b++) {
              if (this.bodies[b]) {
                getPosition(this.simulation.xpos, b, this.bodies[b].position);
                getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
                this.bodies[b].updateWorldMatrix();
              }
            }
            const bodyID = dragged.bodyID;
            this.dragStateManager.update();
            const force = toMujocoPos(
              this.dragStateManager.currentWorld.clone()
                .sub(this.dragStateManager.worldHit)
                .multiplyScalar(60.0)
            );
            // clamp force magnitude
            const forceMagnitude = Math.sqrt(force.x * force.x + force.y * force.y + force.z * force.z);
            const maxForce = 30.0;
            if (forceMagnitude > maxForce) {
              const scale = maxForce / forceMagnitude;
              force.x *= scale;
              force.y *= scale;
              force.z *= scale;
            }
            const point = toMujocoPos(this.dragStateManager.worldHit.clone());
            this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, bodyID);
          }

          this.simulation.step();
        }

        for (let b = 0; b < this.model.nbody; b++) {
          if (!this.bodies[b]) {
            continue;
          }
          if (!this.lastSimState.bodies.has(b)) {
            this.lastSimState.bodies.set(b, {
              position: new THREE.Vector3(),
              quaternion: new THREE.Quaternion()
            });
          }
          const cached = this.lastSimState.bodies.get(b);
          getPosition(this.simulation.xpos, b, cached.position);
          getQuaternion(this.simulation.xquat, b, cached.quaternion);
        }

        const numLights = this.model.nlight;
        for (let l = 0; l < numLights; l++) {
          if (!this.lights[l]) {
            continue;
          }
          if (!this.lastSimState.lights.has(l)) {
            this.lastSimState.lights.set(l, {
              position: new THREE.Vector3(),
              direction: new THREE.Vector3()
            });
          }
          const cached = this.lastSimState.lights.get(l);
          getPosition(this.simulation.light_xpos, l, cached.position);
          getPosition(this.simulation.light_xdir, l, cached.direction);
        }

        this.lastSimState.tendons.numWraps = {
          count: this.model.nwrap,
          matrix: this.lastSimState.tendons.matrix
        };

        this._stepFrameCount += 1;
        const now = performance.now();
        const elapsedStep = now - this._stepLastTime;
        if (elapsedStep >= 500) {
          this.simStepHz = (this._stepFrameCount * 1000) / elapsedStep;
          this._stepFrameCount = 0;
          this._stepLastTime = now;
        }
      } else {
        this.simStepHz = 0;
        this._stepFrameCount = 0;
        this._stepLastTime = performance.now();
      }

      const loopEnd = performance.now();
      const elapsed = (loopEnd - loopStart) / 1000;
      const target = this.timestep * this.decimation;
      const sleepTime = Math.max(0, target - elapsed);
      await new Promise((resolve) => setTimeout(resolve, sleepTime * 1000));
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(this.renderScale);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this._lastRenderTime = 0;
    this.render();
  }

  setRenderScale(scale) {
    const clamped = Math.max(0.5, Math.min(2.0, scale));
    this.renderScale = clamped;
    this.renderer.setPixelRatio(this.renderScale);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this._lastRenderTime = 0;
    this.render();
  }

  getSimStepHz() {
    return this.simStepHz;
  }

  readPolicyState() {
    const qpos = this.simulation.qpos;
    const qvel = this.simulation.qvel;
    const jointPos = new Float32Array(this.numActions);
    const jointVel = new Float32Array(this.numActions);
    for (let i = 0; i < this.numActions; i++) {
      const qposAdr = this.qpos_adr_policy[i];
      const qvelAdr = this.qvel_adr_policy[i];
      jointPos[i] = qpos[qposAdr];
      jointVel[i] = qvel[qvelAdr];
    }
    const rootPos = new Float32Array([qpos[0], qpos[1], qpos[2]]);
    const rootQuat = new Float32Array([qpos[3], qpos[4], qpos[5], qpos[6]]);
    const rootAngVel = new Float32Array([qvel[3], qvel[4], qvel[5]]);
    return {
      jointPos,
      jointVel,
      rootPos,
      rootQuat,
      rootAngVel
    };
  }

  /**
   * Read policy state for a specific robot (multi-robot support).
   * @param {number} robotIndex - robot index (0-based)
   * @returns {Object} {jointPos, jointVel, rootPos, rootQuat, rootAngVel}
   */
  readPolicyStateForRobot(robotIndex = 0) {
    const mapping = this.robotJointMappings?.[robotIndex];
    if (!mapping) {
      // No mapping: fall back to single-robot state read.
      return this.readPolicyState();
    }
    
    const qpos = this.simulation.qpos;
    const qvel = this.simulation.qvel;
    
    // Joint state.
    const jointPos = new Float32Array(mapping.numActions);
    const jointVel = new Float32Array(mapping.numActions);
    for (let i = 0; i < mapping.numActions; i++) {
      jointPos[i] = qpos[mapping.qpos_adr_policy[i]];
      jointVel[i] = qvel[mapping.qvel_adr_policy[i]];
    }
    
    // Root state (use the mapped freejoint addresses).
    const freejointQposAdr = mapping.freejoint_qpos_adr ?? 0;
    const freejointQvelAdr = mapping.freejoint_qvel_adr ?? 0;
    
    const rootPos = new Float32Array([
      qpos[freejointQposAdr + 0],
      qpos[freejointQposAdr + 1],
      qpos[freejointQposAdr + 2]
    ]);
    const rootQuat = new Float32Array([
      qpos[freejointQposAdr + 3],
      qpos[freejointQposAdr + 4],
      qpos[freejointQposAdr + 5],
      qpos[freejointQposAdr + 6]
    ]);
    const rootAngVel = new Float32Array([
      // freejoint qvel: first 3 are linear velocity, last 3 are angular velocity.
      // Keep consistent with single-robot readPolicyState() which uses qvel[3..5].
      qvel[freejointQvelAdr + 3],
      qvel[freejointQvelAdr + 4],
      qvel[freejointQvelAdr + 5]
    ]);
    
    return {
      jointPos,
      jointVel,
      rootPos,
      rootQuat,
      rootAngVel
    };
  }

  resetSimulation() {
    if (!this.simulation) {
      return;
    }
    this.params.paused = true;
    this.simulation.resetData();
    this.simulation.forward();
    // Multi-robot: re-apply initial positions.
    if (this.robotConfigs && this.robotConfigs.length > 1) {
      this.setMultiRobotInitialPositions();
    }
    this.actionTarget = null;
    
    // Detect multi-robot mode and reset runners.
    const isMultiRobot = this.robotJointMappings && this.robotJointMappings.length > 1;
    
    if (isMultiRobot && this.policyRunners) {
      // Multi-robot: reset all policy runners.
      for (let robotIdx = 0; robotIdx < this.policyRunners.length; robotIdx++) {
        if (this.policyRunners[robotIdx]) {
          const state = this.readPolicyStateForRobot(robotIdx);
          this.policyRunners[robotIdx].reset(state);
        }
      }
      this.params.current_motion = 'default';
    } else if (this.policyRunner) {
      // Single-robot: original logic.
      const state = this.readPolicyState();
      this.policyRunner.reset(state);
      this.params.current_motion = 'default';
    }
    this.params.paused = false;
  }

  /**
   * Request a motion change (multi-robot support).
   *
   * Notes:
   * - Primarily for console debugging / scripted control, so we don't expose Vue component methods.
   * - TrackingHelper.requestMotion() is gated: non-default motions are only allowed when
   *   (currentName === 'default' && currentDone === true).
   *   When force=true, we bypass that gate and start from the current state (debug/testing only).
   *
   * @param {string} name - motion name (must exist in tracking.availableMotions())
   * @param {number|null} robotIndex - null: all robots; >=0: specific robot
   * @param {boolean} force - whether to force switching (bypass requestMotion gating)
   * @returns {boolean} whether the motion was accepted (all / single)
   */
  requestMotion(name, robotIndex = null, force = false) {
    const hasMulti = this.robotJointMappings && this.robotJointMappings.length > 1;
    const runners = (hasMulti && Array.isArray(this.policyRunners) && this.policyRunners.length > 0)
      ? this.policyRunners
      : (this.policyRunner ? [this.policyRunner] : []);

    if (!name || typeof name !== 'string' || runners.length === 0) {
      return false;
    }

    const applyToRobot = (idx) => {
      const runner = runners[idx];
      const tracking = runner?.tracking ?? null;
      if (!tracking) {
        return false;
      }
      // Motion must exist.
      if (!tracking.motions || !tracking.motions[name]) {
        return false;
      }
      const state = hasMulti ? this.readPolicyStateForRobot(idx) : this.readPolicyState();

      let accepted = false;
      if (force && typeof tracking._startMotionFromCurrent === 'function') {
        // Forced switch: bypass TrackingHelper.requestMotion() gating.
        tracking._startMotionFromCurrent(name, state);
        accepted = true;
      } else {
        accepted = tracking.requestMotion(name, state);
      }

      if (accepted && Array.isArray(this.robotConfigs) && this.robotConfigs[idx]) {
        this.robotConfigs[idx].motion = name;
      }
      return accepted;
    };

    if (robotIndex === null) {
      let allAccepted = true;
      for (let i = 0; i < runners.length; i++) {
        const ok = applyToRobot(i);
        if (!ok) {
          allAccepted = false;
        }
      }
      if (allAccepted) {
        this.params.current_motion = name;
      }
      return allAccepted;
    }

    if (typeof robotIndex !== 'number' || !Number.isFinite(robotIndex)) {
      return false;
    }
    const idx = Math.max(0, Math.min(Math.floor(robotIndex), runners.length - 1));
    return applyToRobot(idx);
  }

  render() {
    if (!this.model || !this.data || !this.simulation) {
      return;
    }
    const now = performance.now();
    if (now - this._lastRenderTime < 30) {
      return;
    }
    this._lastRenderTime = now;

    // Update camera follow (if enabled).
    if (this.followEnabled) {
      // Let OrbitControls process wheel zoom etc.
      this.controls.update();
      
      // Ensure followBodyId is a valid pelvis body ID.
      if (this.followBodyId === null || this.followBodyId === undefined) {
        this.updateFollowBodyId();
      }
      
      // Track mouse drag / wheel activity so we don't fight user input.
      const isMouseDown = this._isMouseDragging;
      const isWheeling = this._isWheeling;
      
      // Update the target to the robot position (no position constraints).
      if (this.followBodyId !== null && this.followBodyId !== undefined) {
        const cached = this.lastSimState.bodies.get(this.followBodyId);
        if (cached && cached.position) {
          const pos = cached.position;
          // Verify the body name is actually pelvis (not a goal, etc).
          const body = this.bodies[this.followBodyId];
          if (body && body.name && body.name.includes('pelvis')) {
            // Valid pelvis: update target.
            this.followTargetDesired.set(pos.x, pos.y + this.followHeight, pos.z);
            this.followTarget.lerp(this.followTargetDesired, this.followLerp);
            this.controls.target.copy(this.followTarget);
            
            // Only update camera position when the user is not dragging/zooming.
            if (!isMouseDown && !isWheeling) {
              // Use the saved user offset (if any), otherwise the default offset.
              const offset = this._userFollowOffset || this.followOffset;
              this.camera.position.copy(this.controls.target).add(offset);
            } else {
              // User is controlling camera: update the saved offset.
              this._userFollowOffset = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
            }
          }
        }
      }
    } else {
      // Follow disabled: keep current target (user can focus via UI).
      if (!this._cameraTargetInitialized) {
        this.controls.target.set(0, 0.8, 0);
        this._cameraTargetInitialized = true;
      }
    }
    
    // WASDQE keyboard camera translation controls.
    if (this.keys) {
      // Check if any movement key is pressed.
      const isWASDPressed = Object.values(this.keys).some(v => v === true);
      
      const moveSpeed = this.cameraSpeed || 0.5;
      const direction = new THREE.Vector3();
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();
      
      // Compute camera forward/right/up directions.
      this.camera.getWorldDirection(direction);
      right.crossVectors(direction, this.camera.up).normalize();
      up.crossVectors(right, direction).normalize();
      
      let moved = false;
      
      if (this.followEnabled) {
        // Follow enabled: keyboard does nothing; mouse controls via OrbitControls.
      } else {
        // Follow disabled: mouse rotates/zooms; keyboard translates target + camera.
        // Compute translation delta in camera space.
        const moveDelta = new THREE.Vector3();
        
        // W/S: forward/back on the horizontal plane (keep height unchanged).
        const forward = direction.clone();
        forward.y = 0; // project onto horizontal plane
        if (forward.lengthSq() > 0.001) {
          forward.normalize();
          if (this.keys['w']) {
            // W: move forward
            moveDelta.add(forward.clone().multiplyScalar(moveSpeed));
            moved = true;
          }
          if (this.keys['s']) {
            // S: move backward
            moveDelta.add(forward.clone().multiplyScalar(-moveSpeed));
            moved = true;
          }
        }
        
        // A/D: strafe left/right (translate, not rotate).
        if (this.keys['a']) {
          moveDelta.add(right.clone().multiplyScalar(-moveSpeed));
          moved = true;
        }
        if (this.keys['d']) {
          moveDelta.add(right.clone().multiplyScalar(moveSpeed));
          moved = true;
        }
        
        // Q/E: move camera height up/down (Y axis).
        if (this.keys['q']) {
          // Q: move up
          moveDelta.y += moveSpeed;
          moved = true;
        }
        if (this.keys['e']) {
          // E: move down
          moveDelta.y -= moveSpeed;
          moved = true;
        }
        
        // Apply translation to both target and camera (true pan).
        if (moved) {
          this.controls.target.add(moveDelta);
          this.camera.position.add(moveDelta);
          this.controls.update();
        }
      }
    } else {
      // No keys pressed.
      if (this.followEnabled) {
        // Follow enabled: mouse controls everything.
        this.controls.enableRotate = true;
        this.controls.enablePan = true;
        this.controls.enableZoom = true;
      } else {
        // Follow disabled: mouse + keyboard controls.
        this.controls.enableRotate = true; // mouse drag rotate
        this.controls.enablePan = false; // disable mouse pan
        this.controls.enableZoom = true; // mouse wheel zoom
      }
    }
    
    this.controls.update();

    for (const [b, cached] of this.lastSimState.bodies) {
      if (this.bodies[b]) {
        this.bodies[b].position.copy(cached.position);
        this.bodies[b].quaternion.copy(cached.quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    for (const [l, cached] of this.lastSimState.lights) {
      if (this.lights[l]) {
        this.lights[l].position.copy(cached.position);
        this.lights[l].lookAt(cached.direction.clone().add(this.lights[l].position));
      }
    }

    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      const numWraps = this.lastSimState.tendons.numWraps.count;
      this.mujocoRoot.cylinders.count = numWraps;
      this.mujocoRoot.spheres.count = numWraps > 0 ? numWraps + 1 : 0;
      this.mujocoRoot.cylinders.instanceMatrix.needsUpdate = true;
      this.mujocoRoot.spheres.instanceMatrix.needsUpdate = true;
    }

    this.renderer.render(this.scene, this.camera);
  }
}
