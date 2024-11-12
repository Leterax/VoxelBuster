#version 450

const bool USE_BRANCHLESS_DDA = true;
const int MAX_RAY_STEPS = 256;

layout(local_size_x = 16, local_size_y = 16) in;

// Input voxel data buffer
layout(std430, binding = 0) buffer VoxelBuffer {
    int voxels[];  // 1D buffer storing voxel data
};

// Output image
layout(binding = 0, rgba32f) uniform image2D u_imageOutput;

// Uniforms
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cameraPos;
uniform ivec3 u_voxelGridDim;
uniform ivec2 u_screenSize;

// Utility function to calculate the 1D index from 3D voxel coordinates
int getVoxelIndex(ivec3 pos) {
    return pos.x + u_voxelGridDim.x * (pos.y + u_voxelGridDim.y * pos.z);
}

// Function to check if a voxel is filled
bool isVoxelFilled(ivec3 pos) {
    if (any(lessThan(pos, ivec3(0))) || any(greaterThanEqual(pos, u_voxelGridDim))) {
        return false;  // Out of bounds
    }
    return voxels[getVoxelIndex(pos)] > 0;
}

vec2 rotate2d(vec2 v, float a) {
	float sinA = sin(a);
	float cosA = cos(a);
	return vec2(v.x * cosA - v.y * sinA, v.y * cosA + v.x * sinA);	
}



vec4 raymarch(vec3 rayOrigin, vec3 rayDir) {
    vec3 rayPos = rayOrigin;
    ivec3 mapPos = ivec3(floor(rayPos));

    // Calculate distances for ray marching in each axis direction
    vec3 deltaDist = abs(vec3(length(rayDir)) / rayDir);
    
    // Determine the step direction along each axis
    ivec3 rayStep = ivec3(sign(rayDir));
    
    // Calculate initial side distances
    vec3 sideDist = (sign(rayDir) * (vec3(mapPos) - rayPos) + (sign(rayDir) * 0.5) + 0.5) * deltaDist;
    
    // Mask to track the component along which the ray advances
    bvec3 mask;

    // Start ray marching
    for (int i = 0; i < MAX_RAY_STEPS; i++) {
        // Check if the current voxel exists
        if (isVoxelFilled(mapPos)) {
            // Determine color based on last axis moved (for illustration)
            vec3 color = vec3(0.0);
            if (mask.x) color = vec3(0.5);
            if (mask.y) color = vec3(1.0);
            if (mask.z) color = vec3(0.75);

            // Calculate normal from last axis moved
            vec3 normal = vec3(0.0);
            if (mask.x) normal =vec3(1.0, 0.0, 0.0);
            if (mask.y) normal =  vec3(0.0, 1.0, 0.0);
            if (mask.z) normal =  vec3(0.0, 0.0, 1.0);

            float depth = length(rayPos - rayOrigin);
            return vec4(normal, depth); // Optionally return the normal in a different way if needed
        }

        // Branchless DDA approach for determining which axis to step along
        if (USE_BRANCHLESS_DDA) {
            mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
            sideDist += vec3(mask) * deltaDist;
            mapPos += ivec3(vec3(mask)) * rayStep;
        } else {
            // Conditional stepping based on the smallest side distance
            if (sideDist.x < sideDist.y) {
                if (sideDist.x < sideDist.z) {
                    sideDist.x += deltaDist.x;
                    mapPos.x += rayStep.x;
                    mask = bvec3(true, false, false);
                } else {
                    sideDist.z += deltaDist.z;
                    mapPos.z += rayStep.z;
                    mask = bvec3(false, false, true);
                }
            } else {
                if (sideDist.y < sideDist.z) {
                    sideDist.y += deltaDist.y;
                    mapPos.y += rayStep.y;
                    mask = bvec3(false, true, false);
                } else {
                    sideDist.z += deltaDist.z;
                    mapPos.z += rayStep.z;
                    mask = bvec3(false, false, true);
                }
            }
        }
    }

    // Return default color and depth if no voxel is hit within the maximum steps
    return vec4(0.0, 0.0, 0.0, -1.0); // -1.0 for depth to indicate no hit
}


void main() {
    ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
    if (pixelCoords.x >= u_screenSize.x || pixelCoords.y >= u_screenSize.y) {
        return;
    }

    // Compute normalized device coordinates
    vec2 ndc = (vec2(pixelCoords) / vec2(u_screenSize)) * 2.0 - 1.0;
    // ndc.y = -ndc.y;

    // Compute clip space position
    vec4 rayClip = vec4(ndc, -1.0, 1.0);

    // Compute eye space position
    vec4 rayEye = inverse(u_proj) * rayClip;
    rayEye.z = -1.0;
    rayEye.w = 0.0;

    // Compute world space direction
    vec3 rayDir = normalize((inverse(u_view) * rayEye).xyz);
    vec3 rayOrigin = u_cameraPos;

    // Perform raymarching
    vec4 color = raymarch(rayOrigin, rayDir);

    color.a = 1.0;  // Set alpha to 1.0
    // Store the result
    imageStore(u_imageOutput, pixelCoords, color);
}
