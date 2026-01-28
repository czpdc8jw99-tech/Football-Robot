/**
 * Multi-robot scene generator.
 * Generates a MuJoCo XML scene containing multiple robots based on robotConfigs.
 * 
 * Features:
 * - Clone the robot body and all nested elements (joints, geoms, sites, ...)
 * - Rename all elements per robot to avoid name collisions
 * - Clone and rename actuator motors
 * - Set each robot initial position
 * - Supports up to 11 robots
 * 
 * @module multiRobotGenerator
 */

/**
 * Generate a multi-robot XML scene.
 * @param {string} baseXmlPath - base XML path (relative to public/)
 * @param {Array<{x: number, y: number, z: number}>} robotConfigs - per-robot initial positions
 * @returns {Promise<string>} generated XML as a string
 * @throws {Error} if base XML cannot be loaded or pelvis body cannot be found
 */
export async function generateMultiRobotXML(baseXmlPath, robotConfigs) {
  // Load base XML.
  const response = await fetch(baseXmlPath);
  if (!response.ok) {
    throw new Error(`Failed to load base XML: ${response.status}`);
  }
  let xmlContent = await response.text();
  
  // Single robot: just update position.
  if (robotConfigs.length === 1) {
    const config = robotConfigs[0];
    // Update pelvis body position.
    xmlContent = xmlContent.replace(
      /<body name="pelvis" pos="[^"]*"/g,
      `<body name="pelvis" pos="${config.x} ${config.y} ${config.z}"`
    );
    return xmlContent;
  }
  
  // Find pelvis body start/end.
  const pelvisStartMarker = '<body name="pelvis"';
  const pelvisStartIdx = xmlContent.indexOf(pelvisStartMarker);
  if (pelvisStartIdx === -1) {
    throw new Error('Could not find pelvis body in XML');
  }
  
  // Find the end of the pelvis body (need to match nested <body> tags).
  let bodyDepth = 0;
  let pelvisEndIdx = -1;
  let foundPelvisStart = false;
  
  for (let i = pelvisStartIdx; i < xmlContent.length; i++) {
    const substr = xmlContent.substring(i, Math.min(i + 10, xmlContent.length));
    if (substr.startsWith('<body')) {
      bodyDepth++;
      if (i === pelvisStartIdx) {
        foundPelvisStart = true;
      }
    } else if (substr.startsWith('</body>')) {
      bodyDepth--;
      if (foundPelvisStart && bodyDepth === 0) {
        pelvisEndIdx = i + 7; // '</body>'.length = 7
        break;
      }
    }
  }
  
  if (pelvisEndIdx === -1) {
    throw new Error('Could not find end of pelvis body');
  }
  
  // Extract full pelvis body content (including start/end tags).
  const pelvisBodyContent = xmlContent.substring(pelvisStartIdx, pelvisEndIdx);
  
  // Find actuator section.
  const actuatorStartMarker = '<actuator>';
  const actuatorStartIdx = xmlContent.indexOf(actuatorStartMarker);
  if (actuatorStartIdx === -1) {
    throw new Error('Could not find actuator section');
  }
  
  // Find the first motor within <actuator>.
  const motorStartMarker = '<motor name="';
  let motorStartIdx = xmlContent.indexOf(motorStartMarker, actuatorStartIdx);
  if (motorStartIdx === -1) {
    throw new Error('Could not find motors in actuator section');
  }
  
  // Find the end of motors (right before </actuator>).
  const actuatorEndMarker = '</actuator>';
  const actuatorEndIdx = xmlContent.indexOf(actuatorEndMarker);
  if (actuatorEndIdx === -1) {
    throw new Error('Could not find end of actuator section');
  }
  
  // Extract all motor entries.
  const motorsContent = xmlContent.substring(motorStartIdx, actuatorEndIdx);
  
  // Build new XML content.
  let newXmlContent = xmlContent.substring(0, pelvisStartIdx);
  
  // Add one robot body per config.
  robotConfigs.forEach((config, index) => {
    const robotId = index === 0 ? '' : `robot${index + 1}_`;
    const robotBody = addPrefixToRobotBody(pelvisBodyContent, robotId, config);
    newXmlContent += robotBody + '\n        ';
  });
  
  // Append remaining XML (pelvis end -> actuator motors start).
  newXmlContent += xmlContent.substring(pelvisEndIdx, motorStartIdx);
  
  // Add actuator motors per robot.
  robotConfigs.forEach((config, index) => {
    if (index === 0) {
      // Robot 1 uses original motors.
      newXmlContent += motorsContent;
    } else {
      // Other robots use prefixed motors.
      const robotId = `robot${index + 1}_`;
      const robotMotors = addPrefixToMotors(motorsContent, robotId);
      newXmlContent += robotMotors;
    }
  });
  
  // Append remaining XML.
  newXmlContent += xmlContent.substring(actuatorEndIdx);
  
  return newXmlContent;
}

/**
 * Add a prefix to robot body elements and set position (improved).
 */
function addPrefixToRobotBody(bodyContent, prefix, config) {
  let newBody = bodyContent;
  
  // If prefix is empty (robot 1), only update position.
  if (prefix === '') {
    // Update pelvis body position only.
    newBody = newBody.replace(/<body name="pelvis" pos="[^"]*"/g, 
      `<body name="pelvis" pos="${config.x} ${config.y} ${config.z}"`);
    return newBody;
  }
  
  // Step 1: replace all body names (including nested bodies).
  // Use a tighter regex: only match name within <body ...>.
  newBody = newBody.replace(/<body\s+name="([^"]*)"/g, (match, name) => {
    return `<body name="${prefix}${name}"`;
  });
  
  // Step 2: replace all joint names (including freejoint).
  newBody = newBody.replace(/<joint\s+name="([^"]*)"/g, (match, name) => {
    return `<joint name="${prefix}${name}"`;
  });
  // Handle <freejoint> separately.
  newBody = newBody.replace(/<freejoint\s+name="([^"]*)"/g, (match, name) => {
    return `<freejoint name="${prefix}${name}"`;
  });
  
  // Step 3: replace all site names.
  newBody = newBody.replace(/<site\s+name="([^"]*)"/g, (match, name) => {
    return `<site name="${prefix}${name}"`;
  });
  
  // Step 4: replace all geom names (only match <geom ... name="...">; avoid material/mesh names).
  newBody = newBody.replace(/<geom\s+([^>]*\s+)?name="([^"]*)"/g, (match, before, name) => {
    // Skip already-prefixed names.
    if (name.startsWith(prefix)) {
      return match;
    }
    return `<geom ${before || ''}name="${prefix}${name}"`;
  });
  
  // Step 5: set pelvis body position (must run after renaming).
  // Escape prefix because it may contain regex-special characters.
  const escapedPrefix = prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  newBody = newBody.replace(
    new RegExp(`<body name="${escapedPrefix}pelvis" pos="[^"]*"`, 'g'),
    `<body name="${prefix}pelvis" pos="${config.x} ${config.y} ${config.z}"`
  );
  
  return newBody;
}

/**
 * Add a prefix to motors (improved).
 */
function addPrefixToMotors(motorsContent, prefix) {
  let newMotors = motorsContent;
  
  // Replace motor names (only match name within <motor ...>).
  newMotors = newMotors.replace(/<motor\s+name="([^"]*)"/g, (match, name) => {
    return `<motor name="${prefix}${name}"`;
  });
  
  // Replace motor joint references (only match joint within <motor ...>).
  newMotors = newMotors.replace(/<motor\s+([^>]*\s+)?joint="([^"]*)"/g, (match, before, jointName) => {
    // Skip already-prefixed joint names.
    if (jointName.startsWith(prefix)) {
      return match;
    }
    return `<motor ${before || ''}joint="${prefix}${jointName}"`;
  });
  
  return newMotors;
}
