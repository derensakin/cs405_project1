function multiplyMatrices(matrixA, matrixB) {
    var result = [];

    for (var i = 0; i < 4; i++) {
        result[i] = [];
        for (var j = 0; j < 4; j++) {
            var sum = 0;
            for (var k = 0; k < 4; k++) {
                sum += matrixA[i * 4 + k] * matrixB[k * 4 + j];
            }
            result[i][j] = sum;
        }
    }

    // Flatten the result array
    return result.reduce((a, b) => a.concat(b), []);
}
function createIdentityMatrix() {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
}
function createScaleMatrix(scale_x, scale_y, scale_z) {
    return new Float32Array([
        scale_x, 0, 0, 0,
        0, scale_y, 0, 0,
        0, 0, scale_z, 0,
        0, 0, 0, 1
    ]);
}

function createTranslationMatrix(x_amount, y_amount, z_amount) {
    return new Float32Array([
        1, 0, 0, x_amount,
        0, 1, 0, y_amount,
        0, 0, 1, z_amount,
        0, 0, 0, 1
    ]);
}

function createRotationMatrix_Z(radian) {
    return new Float32Array([
        Math.cos(radian), -Math.sin(radian), 0, 0,
        Math.sin(radian), Math.cos(radian), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_X(radian) {
    return new Float32Array([
        1, 0, 0, 0,
        0, Math.cos(radian), -Math.sin(radian), 0,
        0, Math.sin(radian), Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_Y(radian) {
    return new Float32Array([
        Math.cos(radian), 0, Math.sin(radian), 0,
        0, 1, 0, 0,
        -Math.sin(radian), 0, Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function getTransposeMatrix(matrix) {
    return new Float32Array([
        matrix[0], matrix[4], matrix[8], matrix[12],
        matrix[1], matrix[5], matrix[9], matrix[13],
        matrix[2], matrix[6], matrix[10], matrix[14],
        matrix[3], matrix[7], matrix[11], matrix[15]
    ]);
}

const vertexShaderSource = `
attribute vec3 position;
attribute vec3 normal; // Normal vector for lighting

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 normalMatrix;

uniform vec3 lightDirection;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vNormal = vec3(normalMatrix * vec4(normal, 0.0));
    vLightDirection = lightDirection;

    gl_Position = vec4(position, 1.0) * projectionMatrix * modelViewMatrix; 
}

`

const fragmentShaderSource = `
precision mediump float;

uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float shininess;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(vLightDirection);
    
    // Ambient component
    vec3 ambient = ambientColor;

    // Diffuse component
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    // Specular component (view-dependent)
    vec3 viewDir = vec3(0.0, 0.0, 1.0); // Assuming the view direction is along the z-axis
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * specularColor;

    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}

`

/**
 * @WARNING DO NOT CHANGE ANYTHING ABOVE THIS LINE
 */



/**
 * 
 * @TASK1 Calculate the model view matrix by using the chatGPT
 */

function getChatGPTModelViewMatrix() {
    const transformationMatrix = new Float32Array([
        0.1767767, -0.3061862, 0.3535534, 0.3,
        0.4330127, 0.25, -0.1767767, -0.25,
        0.25, 0.4330127, 0.6123724, 0,
        0, 0, 0, 1
    ]);
    
    return getTransposeMatrix(transformationMatrix);
}


/**
 * 
 * @TASK2 Calculate the model view matrix by using the given 
 * transformation methods and required transformation parameters
 * stated in transformation-prompt.txt
 */
function getModelViewMatrix() {
    const degToRad = (degrees) => degrees * (Math.PI / 180);

    // Multiply two 4x4 matrices
    const multiplyMatrices = (a, b) => {
        const result = new Array(16).fill(0);
        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                for (let i = 0; i < 4; i++) {
                    result[row * 4 + col] += a[row * 4 + i] * b[i * 4 + col];
                }
            }
        }
        return result;
    };

    // Translation matrix
    const translationMatrix = (tx, ty, tz) => {
        return [
            1, 0, 0, tx,
            0, 1, 0, ty,
            0, 0, 1, tz,
            0, 0, 0, 1
        ];
    };

    // Scaling matrix
    const scalingMatrix = (sx, sy, sz) => {
        return [
            sx, 0, 0, 0,
            0, sy, 0, 0,
            0, 0, sz, 0,
            0, 0, 0, 1
        ];
    };

    // Rotation matrix around the x-axis
    const rotationXMatrix = (angle) => {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return [
            1, 0, 0, 0,
            0, c, -s, 0,
            0, s, c, 0,
            0, 0, 0, 1
        ];
    };

    // Rotation matrix around the y-axis
    const rotationYMatrix = (angle) => {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return [
            c, 0, s, 0,
            0, 1, 0, 0,
            -s, 0, c, 0,
            0, 0, 0, 1
        ];
    };

    // Rotation matrix around the z-axis
    const rotationZMatrix = (angle) => {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return [
            c, -s, 0, 0,
            s, c, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ];
    };

    // Apply the transformations step by step

    // Translation: 0.3 units in x, -0.25 units in y
    let modelMatrix = translationMatrix(0.3, -0.25, 0);

    // Scaling: 0.5 on x and y axes
    modelMatrix = multiplyMatrices(modelMatrix, scalingMatrix(0.5, 0.5, 1));

    // Rotation: 30 degrees on x-axis, 45 degrees on y-axis, 60 degrees on z-axis
    modelMatrix = multiplyMatrices(modelMatrix, rotationXMatrix(degToRad(30)));
    modelMatrix = multiplyMatrices(modelMatrix, rotationYMatrix(degToRad(45)));
    modelMatrix = multiplyMatrices(modelMatrix, rotationZMatrix(degToRad(60)));

    // Return the result as Float32Array
    return new Float32Array(modelMatrix);
}

// Example usage:
const transformationMatrix = getModelViewMatrix();
console.log(transformationMatrix);


/**
 * 
 * @TASK3 Ask CHAT-GPT to animate the transformation calculated in 
 * task2 infinitely with a period of 10 seconds. 
 * First 5 seconds, the cube should transform from its initial 
 * position to the target position.
 * The next 5 seconds, the cube should return to its initial position.
 */
function getPeriodicMovement(startTime) {
    const currentTime = Date.now();
    const elapsedTime = (currentTime - startTime) % 10000; // 10 seconds cycle
    const t = (elapsedTime < 5000) ? (elapsedTime / 5000) : (2 - (elapsedTime / 5000)); // Normalize t between 0 and 1

    // Interpolation between initial and target transformations
    const positionX = lerp(0, 0.3, t);
    const positionY = lerp(0, -0.25, t);
    const scaleX = lerp(1, 0.5, t);
    const scaleY = lerp(1, 0.5, t);
    const scaleZ = 1; // Assuming no scaling on z-axis
    const rotationX = lerp(0, Math.PI / 6, t); // 30 degrees in radians
    const rotationY = lerp(0, Math.PI / 4, t); // 45 degrees in radians
    const rotationZ = lerp(0, Math.PI / 3, t); // 60 degrees in radians

    // Compute rotation matrices
    const cosX = Math.cos(rotationX), sinX = Math.sin(rotationX);
    const cosY = Math.cos(rotationY), sinY = Math.sin(rotationY);
    const cosZ = Math.cos(rotationZ), sinZ = Math.sin(rotationZ);

    // Rotation matrix (Rz * Ry * Rx)
    const rotationMatrix = [
        cosY * cosZ,                          cosX * sinZ + sinX * sinY * cosZ, sinX * sinZ - cosX * sinY * cosZ, 0,
        -cosY * sinZ,                         cosX * cosZ - sinX * sinY * sinZ, sinX * cosZ + cosX * sinY * sinZ, 0,
        sinY,                                 -sinX * cosY,                     cosX * cosY,                     0,
        0,                                    0,                                0,                                1
    ];

    // Scale matrix
    const scaleMatrix = [
        scaleX, 0,      0,      0,
        0,      scaleY, 0,      0,
        0,      0,      scaleZ, 0,
        0,      0,      0,      1
    ];

    // Translation matrix
    const translationMatrix = [
        1, 0, 0, positionX,
        0, 1, 0, positionY,
        0, 0, 1, 0,
        0, 0, 0, 1
    ];

    // Apply transformations in order: rotation -> scaling -> translation
    const rotationAndScale = multiplyMatrices(rotationMatrix, scaleMatrix);
    const modelViewMatrix = multiplyMatrices(translationMatrix, rotationAndScale);

    return new Float32Array(modelViewMatrix);
}

// Helper function to linearly interpolate between two values
function lerp(a, b, t) {
    return a + t * (b - a);
}

// Helper function to multiply two 4x4 matrices
function multiplyMatrices(a, b) {
    const result = new Array(16);
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            result[i * 4 + j] = 
                a[i * 4 + 0] * b[0 * 4 + j] +
                a[i * 4 + 1] * b[1 * 4 + j] +
                a[i * 4 + 2] * b[2 * 4 + j] +
                a[i * 4 + 3] * b[3 * 4 + j];
        }
    }
    return result;
}




