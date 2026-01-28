<template>
  <div id="mujoco-container"></div>
  <div class="global-alerts">
    <v-alert
      v-if="isSmallScreen"
      v-model="showSmallScreenAlert"
      type="warning"
      variant="flat"
      density="compact"
      closable
      class="small-screen-alert"
    >
      Screen too small. The control panel is unavailable on small screens. Please use a desktop device.
    </v-alert>
    <v-alert
      v-if="isSafari"
      v-model="showSafariAlert"
      type="warning"
      variant="flat"
      density="compact"
      closable
      class="safari-alert"
    >
      Safari has lower memory limits, which can cause WASM to crash.
    </v-alert>
  </div>
  <div v-if="!isSmallScreen" class="controls">
    <v-card class="controls-card">
      <v-card-title>
        Football Robot
      </v-card-title>
      <v-card-text class="py-0 controls-body">
          <v-btn
            href="https://github.com/czpdc8jw99-tech/Football-Robot"
            target="_blank"
            variant="text"
            size="small"
            color="primary"
            class="text-capitalize"
          >
            <v-icon icon="mdi-github" class="mr-1"></v-icon>
            Demo Code
          </v-btn>
          <v-btn
            href="https://github.com/Axellwppr/motion_tracking"
            target="_blank"
            variant="text"
            size="small"
            color="primary"
            class="text-capitalize"
          >
            <v-icon icon="mdi-github" class="mr-1"></v-icon>
            Training Code
          </v-btn>
        <v-divider class="my-2"/>
        <span class="status-name">Policy</span>
        <div v-if="policyDescription" class="text-caption">{{ policyDescription }}</div>
        <v-select
          v-model="currentPolicy"
          :items="policyItems"
          class="mt-2"
          label="Select policy"
          density="compact"
          hide-details
          item-title="title"
          item-value="value"
          :disabled="isPolicyLoading || state !== 1"
          @update:modelValue="onPolicyChange"
        ></v-select>
        <v-progress-linear
          v-if="isPolicyLoading"
          indeterminate
          height="4"
          color="primary"
          class="mt-2"
        ></v-progress-linear>
        <v-alert
          v-if="policyLoadError"
          type="error"
          variant="tonal"
          density="compact"
          class="mt-2"
        >
          {{ policyLoadError }}
        </v-alert>

        <v-divider class="my-2"/>
        <div class="motion-status" v-if="trackingState">
          <div class="status-legend" v-if="trackingState.available">
            <span class="status-name">Current motion: {{ trackingState.currentName }}</span>
          </div>
        </div>

          <v-progress-linear
            v-if="shouldShowProgress"
            :model-value="progressValue"
            height="5"
            color="primary"
            rounded
            class="mt-3 motion-progress-no-animation"
          ></v-progress-linear>
        <template v-if="trackingState && trackingState.available">
          <v-alert
            v-if="showBackToDefault"
            type="info"
            variant="tonal"
            density="compact"
            class="mt-3"
          >
            Motion "{{ trackingState.currentName }}" finished. Return to the default pose before starting another clip.
            <v-btn
              class="back-to-default-btn"
              color="primary"
              block
              density="compact"
              @click="backToDefault"
            >
              Back to default pose
            </v-btn>
          </v-alert>
          <v-select
            v-model="globalMotionTarget"
            :items="globalMotionItems"
            class="mt-3"
            label="Global motion (all robots)"
            density="compact"
            hide-details
            item-title="title"
            item-value="value"
            :disabled="state !== 1 || isGeneratingRobots"
            @update:modelValue="onGlobalMotionChange"
          ></v-select>
          <div
            v-if="globalPendingMotion"
            class="text-caption mt-1"
            style="color: #0D47A1;"
          >
            Pending: {{ globalPendingMotion }} (auto-play after returning to default)
          </div>
        </template>

        <v-alert
          v-else
          type="info"
          variant="tonal"
          density="compact"
        >
          Loading motion presets…
        </v-alert>

        <v-divider class="my-2"/>
        <div class="upload-section">
          <v-btn
            v-if="!showUploadOptions"
            variant="text"
            density="compact"
            color="primary"
            class="upload-toggle"
            @click="showUploadOptions = true"
          >
            Want to use customized motions?
          </v-btn>
          <template v-else>
            <span class="status-name">Custom motions</span>
            <v-file-input
              v-model="motionUploadFiles"
              label="Upload motion JSON"
              density="compact"
              hide-details
              accept=".json,application/json"
              prepend-icon="mdi-upload"
              multiple
              show-size
              :disabled="state !== 1"
              @update:modelValue="onMotionUpload"
            ></v-file-input>
            <div class="text-caption">
              Read <a target="_blank" href="https://github.com/czpdc8jw99-tech/Football-Robot#readme">readme</a> to learn how to create motion JSON files from GMR.<br/>
              Each file should be a single clip (same schema as motions/default.json). File name becomes the motion name (prefixed with [new]). Duplicate names are ignored.
            </div>
            <v-alert
              v-if="motionUploadMessage"
              :type="motionUploadType"
              variant="tonal"
              density="compact"
            >
              {{ motionUploadMessage }}
            </v-alert>
          </template>
        </div>

        <v-divider class="my-2"/>
        
        <!-- Multi-robot setup -->
        <div class="multi-robot-config mt-2">
          <v-expansion-panels>
            <v-expansion-panel>
              <v-expansion-panel-title>
                <v-icon icon="mdi-robot" class="mr-2"></v-icon>
                Multi-robot setup (up to 11)
              </v-expansion-panel-title>
              <v-expansion-panel-text>
                <v-text-field
                  v-model.number="robotCountDraft"
                  type="number"
                  label="Robot count"
                  min="1"
                  max="11"
                  density="compact"
                  hide-details
                  class="mb-2"
                  @update:modelValue="onRobotCountDraftChange"
                ></v-text-field>

                <div class="text-caption mb-2">
                  In scene: {{ robotConfigs.length }}; Draft: {{ robotCountDraft }}
                </div>
                <div
                  class="text-caption mb-2"
                  v-if="getRobotCountPendingText()"
                  style="color: #0D47A1;"
                >
                  {{ getRobotCountPendingText() }}
                </div>
                
                <div v-for="(robot, index) in robotConfigsDraft" :key="index" class="mb-3">
                  <v-card variant="outlined" density="compact">
                    <v-card-title class="text-caption">
                      Robot {{ index + 1 }}
                      <span v-if="index >= robotConfigs.length" class="text-caption" style="opacity: 0.75;">(pending)</span>
                      <span v-else-if="index >= robotCountDraft" class="text-caption" style="opacity: 0.75;">(will remove)</span>
                    </v-card-title>
                    <v-card-text class="py-2">
                      <v-row dense>
                        <v-col cols="6">
                          <v-text-field
                            v-model.number="robot.x"
                            type="number"
                            label="X"
                            density="compact"
                            hide-details
                            step="0.5"
                            min="-20"
                            max="20"
                            @blur="validateCoordinate('x', index, robot)"
                          ></v-text-field>
                        </v-col>
                        <v-col cols="6">
                          <v-text-field
                            v-model.number="robot.y"
                            type="number"
                            label="Y"
                            density="compact"
                            hide-details
                            step="0.5"
                            min="-10"
                            max="10"
                            @blur="validateCoordinate('y', index, robot)"
                          ></v-text-field>
                        </v-col>
                        <v-col cols="12">
                          <div
                            class="text-caption"
                            v-if="getRobotPositionPendingText(index)"
                            style="color: #0D47A1;"
                          >
                            {{ getRobotPositionPendingText(index) }}
                          </div>
                        </v-col>
                        <v-col cols="12" class="mt-2">
                          <v-select
                            v-model="robot.policyPath"
                            :items="policyPathItems"
                            label="Policy (per robot)"
                            density="compact"
                            hide-details
                            :disabled="state !== 1 || isGeneratingRobots || !!robotPolicyLoading[index]"
                            @update:modelValue="onRobotPolicyChange(index, $event)"
                          ></v-select>
                          <div class="text-caption" v-if="robotPolicyErrors[index]" style="color: #B00020;">
                            {{ robotPolicyErrors[index] }}
                          </div>
                          <div class="text-caption" v-else>
                            Current: {{ (robot.policyPath || '').split('/').pop() || '—' }}
                            <span v-if="robotPolicyLoading[index]">(loading...)</span>
                            <span v-if="index >= robotConfigs.length">(applies after generate)</span>
                          </div>
                        </v-col>

                        <v-col cols="12" class="mt-2">
                          <v-select
                            v-model="robot.motion"
                            :items="getRobotMotionItemsDraft(index)"
                            label="Motion (per robot)"
                            density="compact"
                            hide-details
                            :disabled="state !== 1 || isGeneratingRobots || !!robotPolicyLoading[index]"
                            @update:modelValue="onRobotMotionChange(index, $event)"
                          >
                            <template #item="{ props, item }">
                              <v-list-item
                                v-bind="props"
                                @click="onRobotMotionItemClick(index, item)"
                              ></v-list-item>
                            </template>
                          </v-select>
                          <div class="text-caption" v-if="robotMotionErrors[index]" style="color: #B00020;">
                            {{ robotMotionErrors[index] }}
                          </div>
                          <div class="text-caption" v-else>
                            Now: {{ getRobotTrackingState(index).currentName || '—' }}
                            <span v-if="!getRobotTrackingState(index).currentDone">(playing...)</span>
                            <span v-if="index >= robotConfigs.length">(applies after generate)</span>
                          </div>
                          <div
                            class="text-caption"
                            v-if="index < robotConfigs.length && robotPendingMotions[index]"
                            style="color: #0D47A1;"
                          >
                            Pending: {{ robotPendingMotions[index] }} (auto-play after returning to default)
                          </div>
                        </v-col>

                        <v-col cols="12" v-if="getRobotTrackingState(index).available">
                          <v-progress-linear
                            v-if="shouldShowRobotProgress(index)"
                            :model-value="getRobotProgressValue(index)"
                            height="5"
                            color="primary"
                            rounded
                            class="mt-2 motion-progress-no-animation"
                          ></v-progress-linear>

                          <v-alert
                            v-if="showRobotBackToDefault(index)"
                            type="info"
                            variant="tonal"
                            density="compact"
                            class="mt-2"
                          >
                            Motion "{{ getRobotTrackingState(index).currentName }}" finished. Return to default pose before starting another clip.
                            <v-btn
                              class="back-to-default-btn"
                              color="primary"
                              block
                              density="compact"
                              @click="backToDefaultForRobot(index)"
                            >
                              Back to default pose
                            </v-btn>
                          </v-alert>
                        </v-col>
                      </v-row>
                    </v-card-text>
                  </v-card>
                </div>
                
                <v-btn
                  class="wrap-btn"
                  color="primary"
                  block
                  :disabled="state !== 1 || isGeneratingRobots"
                  @click="generateMultiRobotScene"
                >
                  <v-icon icon="mdi-refresh" class="mr-1"></v-icon>
                  <template v-if="isGeneratingRobots">Generating...</template>
                  <template v-else>
                    <span class="wrap-btn-text">
                      Generate multi-robot<br>
                      scene
                    </span>
                  </template>
                </v-btn>
              </v-expansion-panel-text>
            </v-expansion-panel>
          </v-expansion-panels>
        </div>
        
        <v-snackbar
          v-model="coordinateWarning.show"
          :timeout="3000"
          color="warning"
          location="top"
        >
          {{ coordinateWarning.message }}
          <template v-slot:actions>
            <v-btn
              color="white"
              variant="text"
              @click="coordinateWarning.show = false"
            >
              Close
            </v-btn>
          </template>
        </v-snackbar>
        
        <v-divider class="my-2"/>
        <div class="status-legend mt-2">
          <span class="status-name mb-2">Camera</span>
          <v-select
            v-model.number="selectedRobotIndex"
            :items="robotIndexItems"
            class="mt-2"
            label="Robot"
            density="compact"
            hide-details
            :disabled="state !== 1 || robotConfigs.length < 1"
          ></v-select>
          <v-btn
            size="small"
            variant="outlined"
            color="primary"
            :disabled="state !== 1"
            @click="focusAndFollowRobot"
            block
            class="mt-2"
          >
            <v-icon icon="mdi-crosshairs-gps" class="mr-1"></v-icon>
            Focus & Follow
          </v-btn>
        </div>
        <div class="status-legend">
          <span class="status-name">Render scale</span>
          <span class="text-caption">{{ renderScaleLabel }}</span>
          <span class="status-name">Sim Freq</span>
          <span class="text-caption">{{ simStepLabel }}</span>
        </div>
        <v-slider
          v-model="renderScale"
          min="0.5"
          max="2.0"
          step="0.1"
          density="compact"
          hide-details
          @update:modelValue="onRenderScaleChange"
        ></v-slider>
      </v-card-text>
      <v-card-actions>
        <v-btn color="primary" block @click="reset">Reset</v-btn>
      </v-card-actions>
    </v-card>
  </div>
  <v-dialog :model-value="state === 0" persistent max-width="600px" scrollable>
    <v-card title="Loading Simulation Environment">
      <v-card-text>
        <v-progress-linear indeterminate color="primary"></v-progress-linear>
        Loading MuJoCo and ONNX policy, please wait
      </v-card-text>
    </v-card>
  </v-dialog>
  <v-dialog :model-value="state < 0" persistent max-width="600px" scrollable>
    <v-card title="Simulation Environment Loading Error">
      <v-card-text>
        <span v-if="state === -1">
          Unexpected runtime error, please refresh the page.<br />
          {{ extra_error_message }}
        </span>
        <span v-else-if="state === -2">
          Your browser does not support WebAssembly. Please use a recent version of Chrome, Edge, or Firefox.
        </span>
      </v-card-text>
    </v-card>
  </v-dialog>
</template>

<script>
import { MuJoCoDemo } from '@/simulation/main.js';
import loadMujoco from 'mujoco-js';

export default {
  name: 'DemoPage',
  data: () => ({
    state: 0, // 0: loading, 1: running, -1: JS error, -2: wasm unsupported
    extra_error_message: '',
    keydown_listener: null,
    currentMotion: null,
    availableMotions: [],
    trackingState: {
      available: false,
      currentName: 'default',
      currentDone: true,
      refIdx: 0,
      refLen: 0,
      transitionLen: 0,
      motionLen: 0,
      inTransition: false,
      isDefault: true
    },
    trackingTimer: null,
    policies: [
      {
        value: 'g1-tracking-lafan',
        title: 'G1 Tracking (LaFan1)',
        description: 'General tracking policy trained on LaFan1 dataset.',
        policyPath: './examples/checkpoints/g1/tracking_policy_lafan.json',
        onnxPath: './examples/checkpoints/g1/policy_lafan.onnx'
      },
      {
        value: 'g1-tracking-lafan_amass',
        title: 'G1 Tracking (LaFan1&AMASS)',
        description: 'General tracking policy trained on LaFan1 and AMASS datasets.',
        policyPath: './examples/checkpoints/g1/tracking_policy_amass.json',
        onnxPath: './examples/checkpoints/g1/policy_amass.onnx'
      }
    ],
    currentPolicy: 'g1-tracking-lafan_amass',
    policyLabel: '',
    isPolicyLoading: false,
    policyLoadError: '',
    motionUploadFiles: [],
    motionUploadMessage: '',
    motionUploadType: 'success',
    showUploadOptions: false,
    cameraFollowEnabled: false, // disabled by default; no toggle exposed
    renderScale: 2.0,
    simStepHz: 0,
    isSmallScreen: false,
    showSmallScreenAlert: true,
    isSafari: false,
    showSafariAlert: true,
    resize_listener: null,
    // Multi-robot configuration.
    // Draft count is for editing only; it is applied when generating the scene.
    robotCountDraft: 1,
    // Extend robotConfigs to support per-robot policies.
    // - policyPath: per-robot policy config path (defaults to the selected global policy)
    // - motion: per-robot motion name (defaults to 'default')
    robotConfigs: [{
      x: 0,
      y: 0,
      policyPath: './examples/checkpoints/g1/tracking_policy_amass.json',
      motion: 'default'
    }], // Z is fixed to 0.8 and not shown
    // Draft configs (position/policy/motion) edited before generating the scene.
    robotConfigsDraft: [{
      x: 0,
      y: 0,
      policyPath: './examples/checkpoints/g1/tracking_policy_amass.json',
      motion: 'default'
    }],
    isGeneratingRobots: false,
    // Per-robot policy loading states/errors (visible in UI).
    robotPolicyLoading: [false],
    robotPolicyErrors: [''],
    // Per-robot motion errors + per-robot tracking state.
    robotMotionErrors: [''],
    // Pending motion queue (if tracking isn't ready, queue it; once back to default it auto-plays).
    robotPendingMotions: [''],
    // Global motion target (UI only; keep it "unselected" after apply to support re-triggering).
    globalMotionTarget: null,
    // Global pending label (for the global motion controller).
    globalPendingMotion: '',
    robotTrackingStates: [],
    robotAvailableMotions: [],
    // Focus/follow robot selection.
    selectedRobotIndex: 0, // currently selected robot index (0-based)
    // Coordinate range warning snackbar.
    coordinateWarning: {
      show: false,
      message: ''
    }
  }),
  computed: {
    /**
     * Robot index select items.
     * Generates dropdown items based on current robot count.
     * @returns {Array<{title: string, value: number}>} items
     */
    robotIndexItems() {
      const items = [];
      // Focus/select should be based on robots in the generated scene.
      const count = this.robotConfigs?.length || 1;
      for (let i = 0; i < count; i++) {
        items.push({
          title: `Robot ${i + 1}`,
          value: i
        });
      }
      return items;
    },
    shouldShowProgress() {
      const state = this.trackingState;
      if (!state || !state.available) {
        return false;
      }
      if (state.refLen > 1) {
        return true;
      }
      return !state.currentDone || !state.isDefault || state.inTransition;
    },
    progressValue() {
      const state = this.trackingState;
      if (!state || state.refLen <= 0) {
        return 0;
      }
      const value = ((state.refIdx + 1) / state.refLen) * 100;
      return Math.max(0, Math.min(100, value));
    },
    showBackToDefault() {
      const state = this.trackingState;
      return state && state.available && !state.isDefault && state.currentDone;
    },
    showMotionLockedNotice() {
      const state = this.trackingState;
      return state && state.available && !state.isDefault && !state.currentDone;
    },
    showMotionSelect() {
      const state = this.trackingState;
      if (!state || !state.available) {
        return false;
      }
      if (!state.isDefault || !state.currentDone) {
        return false;
      }
      return this.motionItems.some((item) => !item.disabled);
    },
    motionItems() {
      const names = [...this.availableMotions].sort((a, b) => {
        if (a === 'default') {
          return -1;
        }
        if (b === 'default') {
          return 1;
        }
        return a.localeCompare(b);
      });
      return names.map((name) => ({
        title: name.split('_')[0],
        value: name,
        disabled: name === 'default'
      }));
    },
    motionGroups() {
      const items = this.motionItems.filter((item) => item.value !== 'default');
      if (items.length === 0) {
        return [];
      }
      const customized = [];
      const amass = [];
      const lafan = [];

      for (const item of items) {
        const value = item.value.toLowerCase();
        if (value.includes('[new]')) {
          customized.push(item);
        } else if (value.includes('amass')) {
          amass.push(item);
        } else {
          lafan.push(item);
        }
      }

      const groups = [];
      if (lafan.length > 0) {
        groups.push({ title: 'LAFAN1', items: lafan });
      }
      if (amass.length > 0) {
        groups.push({ title: 'AMASS', items: amass });
      }
      if (customized.length > 0) {
        groups.push({ title: 'Customized', items: customized });
      }
      return groups;
    },
    policyItems() {
      return this.policies.map((policy) => ({
        title: policy.title,
        value: policy.value
      }));
    },
    // Use policyPath as the select value so each robot can store the path directly.
    policyPathItems() {
      return this.policies.map((policy) => ({
        title: policy.title,
        value: policy.policyPath
      }));
    },
    selectedPolicy() {
      return this.policies.find((policy) => policy.value === this.currentPolicy) ?? null;
    },
    policyDescription() {
      return this.selectedPolicy?.description ?? '';
    },
    renderScaleLabel() {
      return `${this.renderScale.toFixed(2)}x`;
    },
    simStepLabel() {
      if (!this.simStepHz || !Number.isFinite(this.simStepHz)) {
        return '—';
      }
      return `${this.simStepHz.toFixed(1)} Hz`;
    },
    globalMotionItems() {
      const names = Array.isArray(this.availableMotions) ? this.availableMotions : [];
      const sorted = [...names].sort((a, b) => {
        if (a === 'default') return -1;
        if (b === 'default') return 1;
        return a.localeCompare(b);
      });
      return sorted.map((name) => ({
        title: name,
        value: name
      }));
    }
  },
  methods: {
    // Clicking the currently-selected item restarts the motion.
    _getSelectItemValue(item) {
      // Vuetify slot "item" shape can vary (value/raw/props)
      return item?.value ?? item?.raw?.value ?? item?.props?.value ?? item;
    },
    onRobotMotionItemClick(robotIndex, item) {
      const value = this._getSelectItemValue(item);
      if (!value) return;
      const current = this.robotConfigsDraft?.[robotIndex]?.motion;
      if (value === current) {
        this.onRobotMotionChange(robotIndex, value);
      }
    },
    // Global motion controller (apply to all generated robots).
    onGlobalMotionChange(motionName) {
      if (!motionName || !this.demo) {
        return;
      }
      // Keep the global selector "unselected" so the user can re-trigger the same motion.
      this.globalMotionTarget = null;
      this.globalPendingMotion = motionName === 'default' ? '' : motionName;

      const appliedCount = this.robotConfigs?.length || 0;
      if (appliedCount <= 0) {
        return;
      }

      // Clear existing errors; pending will be shown per robot.
      this.robotMotionErrors = (this.robotMotionErrors || []).map(() => '');

      if (motionName === 'default') {
        this.robotPendingMotions = (this.robotPendingMotions || []).map(() => '');
        // Sync dropdowns to default
        for (let i = 0; i < appliedCount; i++) {
          if (this.robotConfigsDraft?.[i]) {
            this.robotConfigsDraft[i].motion = 'default';
          }
        }
        if (typeof this.demo.requestMotion === 'function') {
          this.demo.requestMotion('default', null, true);
        } else {
          this.requestMotion('default');
        }
        this.updateTrackingState();
        return;
      }

      // Queue target for all robots, then force default now.
      for (let i = 0; i < appliedCount; i++) {
        this.robotPendingMotions[i] = motionName;
        if (this.robotConfigsDraft?.[i]) {
          this.robotConfigsDraft[i].motion = motionName;
        }
      }
      if (typeof this.demo.requestMotion === 'function') {
        this.demo.requestMotion('default', null, true);
      } else {
        this.requestMotion('default');
      }
      this.updateTrackingState();
    },
    // Ensure draft array length matches the desired draft count.
    ensureDraftLength() {
      const appliedCount = this.robotConfigs?.length || 0;
      // Allow draft count < applied count while still rendering applied robots (avoid motion UI becoming non-editable).
      const desired = Math.max(
        appliedCount,
        Math.max(1, Math.min(11, this.robotCountDraft || 1))
      );
      if (!Array.isArray(this.robotConfigsDraft)) {
        this.robotConfigsDraft = [];
      }
      const next = [];
      for (let i = 0; i < desired; i++) {
        const draft = this.robotConfigsDraft[i];
        const applied = this.robotConfigs?.[i];
        const base = draft ?? applied ?? null;
        next.push({
          x: base?.x ?? 0,
          y: base?.y ?? 0,
          policyPath: base?.policyPath || this.getSelectedPolicyPath(),
          motion: base?.motion || 'default'
        });
      }
      this.robotConfigsDraft = next;
      // Keep UI state arrays in sync (loading/errors).
      this.robotPolicyLoading = Array.from({ length: desired }, (_, i) => this.robotPolicyLoading?.[i] ?? false);
      this.robotPolicyErrors = Array.from({ length: desired }, (_, i) => this.robotPolicyErrors?.[i] ?? '');
      this.robotMotionErrors = Array.from({ length: desired }, (_, i) => this.robotMotionErrors?.[i] ?? '');
      this.robotPendingMotions = Array.from({ length: desired }, (_, i) => this.robotPendingMotions?.[i] ?? '');
    },
    // Draft hints for count/position (auto-clear on revert)
    getRobotCountPendingText() {
      const applied = this.robotConfigs?.length || 0;
      const draft = Math.max(1, Math.min(11, this.robotCountDraft || 1));
      if (draft === applied) {
        return '';
      }
      if (draft > applied) {
        return `Pending: will add ${draft - applied} robot(s) (click "Generate multi-robot scene" to apply)`;
      }
      return `Pending: will remove ${applied - draft} robot(s) (click "Generate multi-robot scene" to apply)`;
    },
    getRobotPositionPendingText(robotIndex) {
      const appliedCount = this.robotConfigs?.length || 0;
      const draftCount = Math.max(1, Math.min(11, this.robotCountDraft || 1));
      // Robots that will be removed: no pending position hint.
      if (robotIndex >= draftCount && robotIndex < appliedCount) {
        return '';
      }
      const draft = this.robotConfigsDraft?.[robotIndex] ?? null;
      if (!draft) {
        return '';
      }
      // New robots not generated yet: show "applies after generate".
      if (robotIndex >= appliedCount) {
        const dx = Number(draft.x ?? 0);
        const dy = Number(draft.y ?? 0);
        return `Pending: position X=${dx}, Y=${dy} (applies after generate)`;
      }
      const applied = this.robotConfigs?.[robotIndex] ?? null;
      if (!applied) {
        return '';
      }
      const ax = Number(applied.x ?? 0);
      const ay = Number(applied.y ?? 0);
      const dx = Number(draft.x ?? 0);
      const dy = Number(draft.y ?? 0);
      if (dx === ax && dy === ay) {
        return '';
      }
      return `Pending: position will be X=${dx}, Y=${dy} (applies after generate)`;
    },
    // Get the current policyPath (used for robotConfigs defaults/backfill).
    getSelectedPolicyPath() {
      const selected = this.policies?.find((p) => p.value === this.currentPolicy);
      return selected?.policyPath ?? './examples/checkpoints/g1/tracking_policy_amass.json';
    },
    // Select a policy for a specific robot (reload only that robot).
    async onRobotPolicyChange(robotIndex, policyPath) {
      // Update draft config.
      if (this.robotConfigsDraft && this.robotConfigsDraft[robotIndex]) {
        this.robotConfigsDraft[robotIndex].policyPath = policyPath;
      }
      // On policy change, clear that robot's pending motion (avoid confusion if motion list changes).
      if (this.robotPendingMotions?.[robotIndex]) {
        this.robotPendingMotions[robotIndex] = '';
      }
      if (this.robotMotionErrors?.[robotIndex] === 'Motion not finished. Return to default first.') {
        this.robotMotionErrors[robotIndex] = '';
      }

      if (!this.demo || typeof this.demo.reloadPolicyForRobot !== 'function') {
        return;
      }

      // If the multi-robot scene isn't generated or runners aren't ready, don't reload yet.
      const isMultiRobot = this.demo?.robotJointMappings?.length > 1 && Array.isArray(this.demo?.policyRunners);
      if (!isMultiRobot) {
        return;
      }
      // Only apply immediately to robots that already exist in the generated scene.
      if (robotIndex >= (this.robotConfigs?.length || 0)) {
        return;
      }

      // Clear errors and start loading.
      this.robotPolicyErrors[robotIndex] = '';
      this.robotPolicyLoading[robotIndex] = true;
      try {
        const wasPaused = this.demo.params?.paused ?? false;
        this.demo.params.paused = true;
        const selected = this.policies.find((p) => p.policyPath === policyPath) ?? null;
        await this.demo.reloadPolicyForRobot(robotIndex, policyPath, {
          onnxPath: selected?.onnxPath
        });
        // After policy change, go back to default to avoid confusion from motion list changes.
        if (this.robotConfigs?.[robotIndex]) {
          this.robotConfigs[robotIndex].policyPath = policyPath;
          this.robotConfigs[robotIndex].motion = 'default';
        }
        if (this.robotConfigsDraft?.[robotIndex]) {
          this.robotConfigsDraft[robotIndex].motion = 'default';
        }
        this.robotMotionErrors[robotIndex] = '';
        this.updateTrackingState();
        this.demo.params.paused = wasPaused;
      } catch (e) {
        const msg = e?.message || e?.toString?.() || 'Policy load failed';
        this.robotPolicyErrors[robotIndex] = msg;
        console.error(e);
      } finally {
        this.robotPolicyLoading[robotIndex] = false;
      }
    },

    // Select a motion for a specific robot (per-robot control).
    onRobotMotionChange(robotIndex, motionName) {
      if (!motionName) {
        return;
      }
      // Per-robot override clears global pending label (avoid misleading UI).
      if (this.globalPendingMotion && this.globalPendingMotion !== motionName) {
        this.globalPendingMotion = '';
      }
      if (this.robotConfigsDraft?.[robotIndex]) {
        this.robotConfigsDraft[robotIndex].motion = motionName;
      }
      this.robotMotionErrors[robotIndex] = '';
      // Each selection overwrites any pending motion for that robot.
      if (this.robotPendingMotions?.[robotIndex]) {
        this.robotPendingMotions[robotIndex] = '';
      }

      // In single-robot mode, allow switching via the robot card too (uses global policyRunner).
      if (!this.demo) {
        return;
      }
      const appliedCount = this.robotConfigs?.length || 0;
      if (robotIndex >= appliedCount) {
        // Robot not generated yet: only save draft; do not affect the current simulation.
        return;
      }

      const isMultiRobot = this.demo?.robotJointMappings?.length > 1 && Array.isArray(this.demo?.policyRunners);
      let accepted = false;

      // When switching during playback, force default then auto-enter the selected motion.
      // (no red error; show pending instead)
      const activeTracking = isMultiRobot
        ? (this.demo.policyRunners?.[robotIndex]?.tracking ?? null)
        : (this.demo?.policyRunner?.tracking ?? null);
      const tsNow = activeTracking?.playbackState?.() ?? this.getRobotTrackingState?.(robotIndex) ?? null;
      const notReadyNow = !!tsNow && (!tsNow.isDefault || !tsNow.currentDone);
      if (motionName && motionName !== 'default' && notReadyNow) {
        this.robotPendingMotions[robotIndex] = motionName;
        this.robotMotionErrors[robotIndex] = '';
        // Force return to default immediately
        if (typeof this.demo.requestMotion === 'function') {
          this.demo.requestMotion('default', robotIndex, true);
        } else if (activeTracking) {
          const state = isMultiRobot
            ? this.demo.readPolicyStateForRobot?.(robotIndex)
            : this.demo.readPolicyState?.();
          activeTracking.requestMotion('default', state);
        }
        this.updateTrackingState();
        return;
      }

      if (isMultiRobot) {
        const tracking = this.demo.policyRunners?.[robotIndex]?.tracking ?? null;
        if (tracking) {
          const state = this.demo.readPolicyStateForRobot?.(robotIndex);
          accepted = tracking.requestMotion(motionName, state);
        } else {
          accepted = false;
        }
      } else {
        // Single robot: reuse existing requestMotion logic.
        accepted = this.requestMotion(motionName);
      }

      if (!accepted) {
        const tracking = this.demo?.policyRunner?.tracking ?? this.demo?.policyRunners?.[robotIndex]?.tracking ?? null;
        const ts = tracking?.playbackState?.() ?? this.getRobotTrackingState?.(robotIndex) ?? null;
        // If tracking isn't ready, queue pending motion.
        const trackingNotReady = !ts || ts.available === false;
        if (motionName && motionName !== 'default' && trackingNotReady) {
          this.robotPendingMotions[robotIndex] = motionName;
          this.robotMotionErrors[robotIndex] = '';
        } else {
          this.robotMotionErrors[robotIndex] = 'Motion switch rejected (unknown motion or not allowed).';
        }
      } else {
        // Switch accepted: clear pending.
        this.robotPendingMotions[robotIndex] = '';
      }
      // Only update applied motion on success (avoid UI desync).
      if (accepted && this.robotConfigs?.[robotIndex]) {
        this.robotConfigs[robotIndex].motion = motionName;
      }
      this.updateTrackingState();
    },

    // Return a specific robot to the default pose.
    backToDefaultForRobot(robotIndex) {
      if (!this.demo || !Array.isArray(this.demo.policyRunners) || !this.demo.policyRunners[robotIndex]) {
        return;
      }
      const tracking = this.demo.policyRunners[robotIndex]?.tracking ?? null;
      if (!tracking) {
        return;
      }
      const state = this.demo.readPolicyStateForRobot?.(robotIndex);
      const accepted = tracking.requestMotion('default', state);
      if (accepted && this.robotConfigs?.[robotIndex]) {
        this.robotConfigs[robotIndex].motion = 'default';
      }
      this.robotMotionErrors[robotIndex] = '';
      this.updateTrackingState();
    },
    detectSafari() {
      const ua = navigator.userAgent;
      return /Safari\//.test(ua)
        && !/Chrome\//.test(ua)
        && !/Chromium\//.test(ua)
        && !/Edg\//.test(ua)
        && !/OPR\//.test(ua)
        && !/SamsungBrowser\//.test(ua)
        && !/CriOS\//.test(ua)
        && !/FxiOS\//.test(ua);
    },
    updateScreenState() {
      const isSmall = window.innerWidth < 500 || window.innerHeight < 700;
      if (!isSmall && this.isSmallScreen) {
        this.showSmallScreenAlert = true;
      }
      this.isSmallScreen = isSmall;
    },
    async init() {
      if (typeof WebAssembly !== 'object' || typeof WebAssembly.instantiate !== 'function') {
        this.state = -2;
        return;
      }

      try {
        const mujoco = await loadMujoco();
        this.demo = new MuJoCoDemo(mujoco);
        // Expose the demo object on window for console debugging.
        window.demo = this.demo;
        // Always keep follow disabled here (UI manages follow).
        this.demo.setFollowEnabled?.(false);
        await this.demo.init();
        this.demo.main_loop();
        this.demo.params.paused = false;
        this.reapplyCustomMotions();
        this.availableMotions = this.getAvailableMotions();
        // Keep global motion selector independent from per-robot state.
        this.globalMotionTarget = null;
        this.globalPendingMotion = '';
        this.currentMotion = this.demo.params.current_motion ?? this.availableMotions[0] ?? null;
        this.startTrackingPoll();
        this.renderScale = this.demo.renderScale ?? this.renderScale;
        const matchingPolicy = this.policies.find(
          (policy) => policy.policyPath === this.demo.currentPolicyPath
        );
        if (matchingPolicy) {
          this.currentPolicy = matchingPolicy.value;
        }
        this.policyLabel = this.demo.currentPolicyPath?.split('/').pop() ?? this.policyLabel;
        this.state = 1;
      } catch (error) {
        this.state = -1;
        this.extra_error_message = error.toString();
        console.error(error);
      }
    },
    reapplyCustomMotions() {
      if (!this.demo || !this.customMotions) {
        return;
      }
      const names = Object.keys(this.customMotions);
      if (names.length === 0) {
        return;
      }
      this.addMotions(this.customMotions);
    },
    async onMotionUpload(files) {
      const fileList = Array.isArray(files)
        ? files
        : files instanceof FileList
          ? Array.from(files)
          : files
            ? [files]
            : [];
      if (fileList.length === 0) {
        return;
      }
      if (!this.demo) {
        this.motionUploadMessage = 'Demo not ready yet. Please wait for loading to finish.';
        this.motionUploadType = 'warning';
        this.motionUploadFiles = [];
        return;
      }

      let added = 0;
      let skipped = 0;
      let invalid = 0;
      let failed = 0;
      const prefix = '[new] ';

      for (const file of fileList) {
        try {
          const text = await file.text();
          const parsed = JSON.parse(text);
          const clip = parsed && typeof parsed === 'object' && !Array.isArray(parsed)
            ? parsed
            : null;
          if (!clip) {
            invalid += 1;
            continue;
          }

          const baseName = file.name.replace(/\.[^/.]+$/, '').trim();
          const normalizedName = baseName ? baseName : 'motion';
          const motionName = normalizedName.startsWith(prefix)
            ? normalizedName
            : `${prefix}${normalizedName}`;
          const result = this.addMotions({ [motionName]: clip });
          added += result.added.length;
          skipped += result.skipped.length;
          invalid += result.invalid.length;

          if (result.added.length > 0) {
            if (!this.customMotions) {
              this.customMotions = {};
            }
            for (const name of result.added) {
              this.customMotions[name] = clip;
            }
          }
        } catch (error) {
          console.error('Failed to read motion JSON:', error);
          failed += 1;
        }
      }

      if (added > 0) {
        this.availableMotions = this.getAvailableMotions();
      }

      const parts = [];
      if (added > 0) {
        parts.push(`Added ${added} motion${added === 1 ? '' : 's'}`);
      }
      if (skipped > 0) {
        parts.push(`Skipped ${skipped} duplicate${skipped === 1 ? '' : 's'}`);
      }
      const badCount = invalid + failed;
      if (badCount > 0) {
        parts.push(`Ignored ${badCount} invalid file${badCount === 1 ? '' : 's'}`);
      }
      if (parts.length === 0) {
        this.motionUploadMessage = 'No motions were added.';
        this.motionUploadType = 'info';
      } else {
        this.motionUploadMessage = `${parts.join('. ')}.`;
        this.motionUploadType = badCount > 0 ? 'warning' : 'success';
      }
      this.motionUploadFiles = [];
    },
    // Focus and enable follow; pressing WASDQE will automatically disable follow (handled in MuJoCoDemo).
    focusAndFollowRobot() {
      if (!this.demo) {
        return;
      }
      const robotIndex = this.selectedRobotIndex || 0;
      if (this.demo.focusOnRobot) {
        this.demo.focusOnRobot(robotIndex);
      }
      if (typeof this.demo.setFollowRobotIndex === 'function') {
        this.demo.setFollowRobotIndex(robotIndex);
      }
      if (typeof this.demo.setFollowEnabled === 'function') {
        this.demo.setFollowEnabled(true);
      }
    },
    // Multi-robot config: clamp and apply draft count.
    onRobotCountDraftChange() {
      this.robotCountDraft = Math.max(1, Math.min(11, this.robotCountDraft || 1));
      this.ensureDraftLength();
    },
    // Validate and clamp coordinate values.
    validateCoordinate(axis, robotIndex, robot) {
      const limits = {
        x: { min: -20, max: 20 },
        y: { min: -10, max: 10 }
      };
      
      const limit = limits[axis];
      const oldValue = robot[axis];
      let newValue = oldValue;
      let message = '';
      
      // Clamp out-of-range values.
      if (oldValue === null || oldValue === undefined || isNaN(oldValue)) {
        newValue = 0;
        message = `Robot ${robotIndex + 1} ${axis.toUpperCase()} is invalid; set to 0.`;
      } else if (oldValue < limit.min) {
        newValue = limit.min;
        message = `Robot ${robotIndex + 1} ${axis.toUpperCase()} ${oldValue} < ${limit.min}; clamped to ${limit.min}.`;
      } else if (oldValue > limit.max) {
        newValue = limit.max;
        message = `Robot ${robotIndex + 1} ${axis.toUpperCase()} ${oldValue} > ${limit.max}; clamped to ${limit.max}.`;
      }
      
      // If a value changed, update and show a warning.
      if (newValue !== oldValue) {
        robot[axis] = newValue;
        this.coordinateWarning.message = message;
        this.coordinateWarning.show = true;
      }
    },
    /**
     * Generate a multi-robot scene.
     * Uses the draft robot count/positions to generate a MuJoCo scene,
     * then reloads scene and policies.
     */
    async generateMultiRobotScene() {
      if (!this.demo || this.isGeneratingRobots) {
        return;
      }
      
      this.isGeneratingRobots = true;
      try {
        // Apply draft config (count/position/policy/motion).
        const desiredCount = Math.max(1, Math.min(11, this.robotCountDraft || 1));
        this.ensureDraftLength();
        const nextConfigs = (this.robotConfigsDraft || []).slice(0, desiredCount).map((c) => ({
          x: c?.x ?? 0,
          y: c?.y ?? 0,
          policyPath: c?.policyPath || this.getSelectedPolicyPath(),
          // After regenerating a scene, always return to default (avoid stale motion state).
          motion: 'default'
        }));

        // Use user-provided XY and add fixed Z.
        const configsWithZ = nextConfigs.map((config) => ({
          ...config,
          x: Math.max(-20, Math.min(20, config.x || 0)), // clamp X
          y: Math.max(-10, Math.min(10, config.y || 0)), // clamp Y
          z: 0.8, // fixed Z
          policyPath: config.policyPath || this.getSelectedPolicyPath(),
          motion: config.motion || 'default'
        }));
        
        // Ask the demo to generate a multi-robot scene.
        await this.demo.generateMultiRobotScene(configsWithZ);
        // Reload the scene.
        await this.demo.reloadScene('g1/g1.xml');
        // Apply initial positions.
        if (configsWithZ.length > 1) {
          this.demo.setMultiRobotInitialPositions();
        }
        // Sync applied configs only after scene generation succeeds.
        this.robotConfigs = nextConfigs;
        this.robotConfigsDraft = nextConfigs.map((c) => ({ ...c }));
        this.robotCountDraft = desiredCount;
        // Clear stale motion errors after generating a new scene.
        this.robotMotionErrors = Array.from({ length: desiredCount }, () => '');
        // Clear pending motions after generating a new scene (new scene starts at default).
        this.robotPendingMotions = Array.from({ length: desiredCount }, () => '');
        // Ensure selectedRobotIndex remains in bounds.
        if (this.selectedRobotIndex >= desiredCount) {
          this.selectedRobotIndex = 0;
        }
        // Reload policy (for each robot).
        await this.onPolicyChange(this.currentPolicy);
      } catch (error) {
        console.error('Failed to generate multi-robot scene:', error);
        const errorMsg = error?.message || error?.toString() || 'Unknown error';
        alert('Failed to generate multi-robot scene: ' + errorMsg);
      } finally {
        this.isGeneratingRobots = false;
      }
    },
    onMotionChange(value) {
      if (!this.demo) {
        return;
      }
      if (!value || value === this.demo.params.current_motion) {
        this.currentMotion = this.demo.params.current_motion ?? value;
        return;
      }
      const accepted = this.requestMotion(value);
      if (!accepted) {
        this.currentMotion = this.demo.params.current_motion;
      } else {
        this.currentMotion = value;
        this.updateTrackingState();
      }
    },
    async onPolicyChange(value) {
      if (!this.demo || !value) {
        return;
      }
      // Policy change can alter motions/state: clear pending motions and stale hints first.
      this.robotPendingMotions = (this.robotPendingMotions || []).map(() => '');
      this.robotMotionErrors = (this.robotMotionErrors || []).map((msg) =>
        msg === 'Motion not finished. Return to default first.' ? '' : msg
      );
      const selected = this.policies.find((policy) => policy.value === value);
      if (!selected) {
        return;
      }
      const needsReload = selected.policyPath !== this.demo.currentPolicyPath || selected.onnxPath;
      if (!needsReload) {
        return;
      }
      const wasPaused = this.demo.params?.paused ?? false;
      this.demo.params.paused = true;
      this.isPolicyLoading = true;
      this.policyLoadError = '';
      try {
        await this.demo.reloadPolicy(selected.policyPath, {
          onnxPath: selected.onnxPath || undefined
        });
        this.policyLabel = selected.policyPath?.split('/').pop() ?? this.policyLabel;
        this.reapplyCustomMotions();
        this.availableMotions = this.getAvailableMotions();
        // Reset global motion selector on policy change.
        this.globalMotionTarget = null;
        this.globalPendingMotion = '';
        this.currentMotion = this.demo.params.current_motion ?? this.availableMotions[0] ?? null;
        this.updateTrackingState();
      } catch (error) {
        console.error('Failed to reload policy:', error);
        this.policyLoadError = error.toString();
      } finally {
        this.isPolicyLoading = false;
        this.demo.params.paused = wasPaused;
      }
    },
    reset() {
      if (!this.demo) {
        return;
      }
      this.demo.resetSimulation();

      // On reset, clear errors and return motions to default (keep UI consistent with simulation).
      this.policyLoadError = '';
      this.motionUploadMessage = '';
      this.motionUploadType = 'success';
      this.robotPolicyErrors = (this.robotPolicyErrors || []).map(() => '');
      this.robotMotionErrors = (this.robotMotionErrors || []).map(() => '');
      this.robotPendingMotions = (this.robotPendingMotions || []).map(() => '');

      if (Array.isArray(this.robotConfigs)) {
        for (const cfg of this.robotConfigs) {
          if (cfg) cfg.motion = 'default';
        }
      }
      if (Array.isArray(this.robotConfigsDraft)) {
        for (const cfg of this.robotConfigsDraft) {
          if (cfg) cfg.motion = 'default';
        }
      }

      this.availableMotions = this.getAvailableMotions();
      this.currentMotion = 'default';
      this.updateTrackingState();
    },
    backToDefault() {
      if (!this.demo) {
        return;
      }
      const accepted = this.requestMotion('default');
      if (accepted) {
        this.currentMotion = 'default';
        this.updateTrackingState();
      }
    },
    startTrackingPoll() {
      this.stopTrackingPoll();
      this.updateTrackingState();
      this.updatePerformanceStats();
      this.trackingTimer = setInterval(() => {
        this.updateTrackingState();
        this.updatePerformanceStats();
      }, 33);
    },
    stopTrackingPoll() {
      if (this.trackingTimer) {
        clearInterval(this.trackingTimer);
        this.trackingTimer = null;
      }
    },
    updateTrackingState() {
      const tracking = this.demo?.policyRunner?.tracking ?? null;
      if (!tracking) {
        this.trackingState = {
          available: false,
          currentName: 'default',
          currentDone: true,
          refIdx: 0,
          refLen: 0,
          transitionLen: 0,
          motionLen: 0,
          inTransition: false,
          isDefault: true
        };
        this.robotTrackingStates = [];
        this.robotAvailableMotions = [];
        return;
      }
      const state = tracking.playbackState();
      this.trackingState = { ...state };
      this.availableMotions = tracking.availableMotions();
      const current = this.demo.params.current_motion ?? state.currentName ?? null;
      if (current && this.currentMotion !== current) {
        this.currentMotion = current;
      }

      // Multi-robot: sync per-robot tracking state + motion lists (UI-visible feedback).
      const isMultiRobot = this.demo?.robotJointMappings?.length > 1 && Array.isArray(this.demo?.policyRunners);
      if (isMultiRobot) {
        const runners = this.demo.policyRunners;
        const states = [];
        const motions = [];
        for (let i = 0; i < runners.length; i++) {
          const t = runners[i]?.tracking ?? null;
          if (!t) {
            states[i] = {
              available: false,
              currentName: 'default',
              currentDone: true,
              refIdx: 0,
              refLen: 0,
              transitionLen: 0,
              motionLen: 0,
              inTransition: false,
              isDefault: true
            };
            motions[i] = [];
            continue;
          }
          const st = t.playbackState();
          states[i] = { ...st };
          motions[i] = t.availableMotions();
          // Keep robotConfigs[i].motion in sync for display only (does not affect policy).
          if (this.robotConfigs?.[i] && st?.currentName) {
            this.robotConfigs[i].motion = st.currentName;
          }
        }
        this.robotTrackingStates = states;
        this.robotAvailableMotions = motions;

        // Auto-apply pending motions when back to default + done.
        const appliedCount = this.robotConfigs?.length || 0;
        for (let i = 0; i < Math.min(appliedCount, states.length); i++) {
          const pending = this.robotPendingMotions?.[i] ?? '';
          const st = states[i];
          // If default is finished, clear stale "not finished" errors.
          if (st && st.available && st.isDefault && st.currentDone && this.robotMotionErrors?.[i] === 'Motion not finished. Return to default first.') {
            this.robotMotionErrors[i] = '';
          }
          if (!pending) continue;
          if (!st || !st.available) continue;
          if (!st.isDefault || !st.currentDone) continue;
          const t = runners[i]?.tracking ?? null;
          if (!t) continue;

          const stateForRobot = this.demo.readPolicyStateForRobot?.(i);
          const accepted = t.requestMotion(pending, stateForRobot);
          if (accepted) {
            this.robotPendingMotions[i] = '';
            this.robotMotionErrors[i] = '';
          } else {
            // Avoid infinite retries: clear pending on first failure and show an error.
            this.robotPendingMotions[i] = '';
            this.robotMotionErrors[i] = 'Pending motion failed to play (unknown motion or not allowed).';
          }
        }
        // Clear global pending label once all robots finished applying it.
        if (this.globalPendingMotion) {
          const stillPending = (this.robotPendingMotions || []).some((m) => m === this.globalPendingMotion);
          if (!stillPending) {
            this.globalPendingMotion = '';
          }
        }
      } else {
        // Single-robot mode still populates robotTrackingStates[0] for UI ("Now:" field).
        this.robotTrackingStates = [ { ...this.trackingState } ];
        this.robotAvailableMotions = [ Array.isArray(this.availableMotions) ? [...this.availableMotions] : [] ];

        // Single-robot mode also supports pending motions (for the robot card motion dropdown).
        const pending = this.robotPendingMotions?.[0] ?? '';
        const st = this.trackingState;
        if (st && st.available && st.isDefault && st.currentDone && this.robotMotionErrors?.[0] === 'Motion not finished. Return to default first.') {
          this.robotMotionErrors[0] = '';
        }
        if (pending && st && st.available && st.isDefault && st.currentDone) {
          const accepted = this.requestMotion(pending);
          if (accepted) {
            this.robotPendingMotions[0] = '';
            this.robotMotionErrors[0] = '';
          } else {
            this.robotPendingMotions[0] = '';
            this.robotMotionErrors[0] = 'Pending motion failed to play (unknown motion or not allowed).';
          }
        }
      }
    },

    // Per-robot tracking state / motion items / progress helpers.
    getRobotTrackingState(robotIndex) {
      const fallback = {
        available: false,
        currentName: 'default',
        currentDone: true,
        refIdx: 0,
        refLen: 0,
        transitionLen: 0,
        motionLen: 0,
        inTransition: false,
        isDefault: true
      };
      return this.robotTrackingStates?.[robotIndex] ?? fallback;
    },
    getRobotMotionItems(robotIndex) {
      const names = this.robotAvailableMotions?.[robotIndex] ?? [];
      const sorted = [...names].sort((a, b) => {
        if (a === 'default') return -1;
        if (b === 'default') return 1;
        return a.localeCompare(b);
      });
      return sorted.map((name) => ({
        title: name,
        value: name,
        disabled: false
      }));
    },
    // Draft robot motion list
    getRobotMotionItemsDraft(robotIndex) {
      const appliedCount = this.robotConfigs?.length || 0;
      const isMultiRobot = this.demo?.robotJointMappings?.length > 1 && Array.isArray(this.demo?.policyRunners);
      // In single-robot mode, use global availableMotions for robot 0.
      if (!isMultiRobot && robotIndex === 0 && appliedCount === 1) {
        const names = this.availableMotions ?? [];
        const sorted = [...names].sort((a, b) => {
          if (a === 'default') return -1;
          if (b === 'default') return 1;
          return a.localeCompare(b);
        });
        return sorted.map((name) => ({
          title: name,
          value: name,
          disabled: false
        }));
      }

      if (robotIndex < appliedCount) {
        return this.getRobotMotionItems(robotIndex);
      }
      // Pending robots start from default; only allow default.
      return [{
        title: 'default',
        value: 'default',
        disabled: false
      }];
    },
    shouldShowRobotProgress(robotIndex) {
      const state = this.getRobotTrackingState(robotIndex);
      if (!state || !state.available) return false;
      if (state.refLen > 1) return true;
      return !state.currentDone || !state.isDefault || state.inTransition;
    },
    getRobotProgressValue(robotIndex) {
      const state = this.getRobotTrackingState(robotIndex);
      if (!state || state.refLen <= 0) return 0;
      const value = ((state.refIdx + 1) / state.refLen) * 100;
      return Math.max(0, Math.min(100, value));
    },
    showRobotBackToDefault(robotIndex) {
      const state = this.getRobotTrackingState(robotIndex);
      return state && state.available && !state.isDefault && state.currentDone;
    },
    showRobotMotionLockedNotice(robotIndex) {
      const state = this.getRobotTrackingState(robotIndex);
      return state && state.available && !state.isDefault && !state.currentDone;
    },
    updatePerformanceStats() {
      if (!this.demo) {
        this.simStepHz = 0;
        return;
      }
      this.simStepHz = this.demo.getSimStepHz?.() ?? this.demo.simStepHz ?? 0;
    },
    onRenderScaleChange(value) {
      if (!this.demo) {
        return;
      }
      this.demo.setRenderScale(value);
    },
    getAvailableMotions() {
      const tracking = this.demo?.policyRunner?.tracking ?? null;
      return tracking ? tracking.availableMotions() : [];
    },
    addMotions(motions, options = {}) {
      const tracking = this.demo?.policyRunner?.tracking ?? null;
      if (!tracking) {
        return { added: [], skipped: [], invalid: [] };
      }
      return tracking.addMotions(motions, options);
    },
    requestMotion(name, robotIndex = null) {
      // Support setting motion for a specific robot.
      // robotIndex === null: apply to all robots (backward compatible)
      // robotIndex >= 0: apply to one robot
      
      const isMultiRobot = this.demo?.robotJointMappings?.length > 1;
      
      if (isMultiRobot && this.demo?.policyRunners) {
        // Multi-robot mode.
        if (robotIndex === null) {
          // Apply to all robots.
          let allAccepted = true;
          for (let robotIdx = 0; robotIdx < this.demo.policyRunners.length; robotIdx++) {
            const tracking = this.demo.policyRunners[robotIdx]?.tracking;
            if (!tracking) {
              continue;
            }
            const state = this.demo.readPolicyStateForRobot(robotIdx);
            const accepted = tracking.requestMotion(name, state);
            if (!accepted) {
              allAccepted = false;
            }
          }
          if (allAccepted) {
            this.demo.params.current_motion = name;
          }
          return allAccepted;
        } else {
          // Apply to one robot.
          const tracking = this.demo.policyRunners[robotIndex]?.tracking;
          if (!tracking) {
            return false;
          }
          const state = this.demo.readPolicyStateForRobot(robotIndex);
          const accepted = tracking.requestMotion(name, state);
          if (accepted) {
            // Update that robot's motion (if robotConfigs has a motion field).
            if (this.demo.robotConfigs && this.demo.robotConfigs[robotIndex]) {
              this.demo.robotConfigs[robotIndex].motion = name;
            }
            // If there's only one robot, also update the global motion.
            if (this.demo.policyRunners.length === 1) {
              this.demo.params.current_motion = name;
            }
          }
          return accepted;
        }
      } else {
        // Single-robot mode: original logic.
        const tracking = this.demo?.policyRunner?.tracking ?? null;
        if (!tracking || !this.demo) {
          return false;
        }
        const state = this.demo.readPolicyState();
        const accepted = tracking.requestMotion(name, state);
        if (accepted) {
          this.demo.params.current_motion = name;
        }
        return accepted;
      }
    }
  },
  mounted() {
    this.customMotions = {};
    this.isSafari = this.detectSafari();
    this.updateScreenState();
    this.resize_listener = () => {
      this.updateScreenState();
    };
    window.addEventListener('resize', this.resize_listener);
    this.init();
    // Initialize drafts (keep draft configs aligned with applied configs).
    this.robotCountDraft = this.robotConfigs?.length || 1;
    this.robotConfigsDraft = (this.robotConfigs || []).map((c) => ({ ...c }));
    this.ensureDraftLength();
    this.keydown_listener = (event) => {
      if (event.code === 'Backspace') {
        this.reset();
      }
    };
    document.addEventListener('keydown', this.keydown_listener);
  },
  beforeUnmount() {
    this.stopTrackingPoll();
    document.removeEventListener('keydown', this.keydown_listener);
    if (this.resize_listener) {
      window.removeEventListener('resize', this.resize_listener);
    }
  }
};
</script>

<style scoped>
.controls {
  position: fixed;
  top: 20px;
  right: 20px;
  width: 320px;
  z-index: 1000;
}

.global-alerts {
  position: fixed;
  top: 20px;
  left: 16px;
  right: 16px;
  max-width: 520px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 1200;
}

.small-screen-alert {
  width: 100%;
}

.safari-alert {
  width: 100%;
}

.controls-card {
  max-height: calc(100vh - 40px);
}

.controls-body {
  max-height: calc(100vh - 160px);
  overflow-y: auto;
  overscroll-behavior: contain;
}

.motion-status {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.motion-groups {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 12px;
  max-height: 200px;
  overflow-y: auto;
}

.motion-group {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.motion-chip {
  text-transform: none;
  font-size: 0.7rem;
}

.status-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.status-name {
  font-weight: 600;
}

.policy-file {
  display: block;
  margin-top: 4px;
}


.upload-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.upload-toggle {
  padding: 0;
  min-height: unset;
  font-size: 0.85rem;
  text-transform: none;
}

.motion-progress-no-animation,
.motion-progress-no-animation *,
.motion-progress-no-animation::before,
.motion-progress-no-animation::after {
  transition: none !important;
  animation: none !important;
}

.motion-progress-no-animation :deep(.v-progress-linear__determinate),
.motion-progress-no-animation :deep(.v-progress-linear__indeterminate),
.motion-progress-no-animation :deep(.v-progress-linear__background) {
  transition: none !important;
  animation: none !important;
}

:deep(.back-to-default-btn) {
  height: auto;
  min-height: 44px;
  padding-top: 6px;
  padding-bottom: 6px;
}

:deep(.back-to-default-btn .v-btn__content) {
  white-space: normal;
  line-height: 1.15;
  text-align: center;
  display: block;
}

::deep(.wrap-btn) {
  height: auto;
  min-height: 44px;
  padding-top: 6px;
  padding-bottom: 6px;
}

::deep(.wrap-btn .v-btn__content) {
  white-space: normal;
  line-height: 1.15;
  text-align: center;
  display: block;
}

::deep(.wrap-btn) {
  height: auto;
  min-height: 44px;
  padding-top: 6px;
  padding-bottom: 6px;
}

::deep(.wrap-btn .v-btn__content) {
  white-space: normal !important;
  line-height: 1.15;
  text-align: center;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  word-break: break-word;
}

::deep(.wrap-btn-text) {
  display: inline-block;
  white-space: normal !important;
  line-height: 1.15;
}
</style>
