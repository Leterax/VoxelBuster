#version 430

#define SIZE_X 1024
#define SIZE_Y 1024
#define SIZE_Z 1024

layout(local_size_x = 32, local_size_y = 32) in;

layout(std430, binding = 0) buffer BlockData {
    uint blocks[];
};

layout(rgba32f, binding = 1) uniform image2D img_output;
layout(r32f, binding = 2) uniform image2D depth_output;

// Camera ubo
layout(binding=3, std140) uniform Camera
{
    vec4 eye, lower_left_corner, horizontal, vertical, origin;
    vec4 u,v,w;
    float lens_radius;
} camera;


uint get_voxel(vec3 position) {
    if (position.x < 0.0 || position.y < 0.0 || position.z < 0.0) {return 0;}
    if (position.x >= SIZE_X || position.y >= SIZE_Y || position.z >= SIZE_Z) {return 0;}
    uint index = uint(position.x) + uint(position.y)*SIZE_X + uint(position.z)*SIZE_X*SIZE_Y;
    if (index >= SIZE_X*SIZE_Y*SIZE_Z) {return 0;}
    return blocks[index];
}


// Utility function to determine the sign of a value
int sign(float x) {
    return x > 0.0 ? 1 : (x < 0.0 ? -1 : 0);
}



// array of colors to pick from
const vec3 colors[8] = vec3[8](
    vec3(0.216, 0.208, 0.065),
    vec3(0.568, 0.044, 0.481),
    vec3(0.630, 0.120, 0.783),
    vec3(0.225, 0.491, 0.756),
    vec3(0.029, 0.134, 0.336),
    vec3(0.168, 0.519, 0.215),
    vec3(0.603, 0.167, 0.099),
    vec3(0.852, 0.887, 0.704)
);


void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(img_output);
    if (pix.x >= size.x || pix.y >= size.y) {return;}
    vec2 uv = vec2(pix)/ vec2(size.x, size.y);

    vec3 dir = normalize(camera.lower_left_corner.xyz + uv.x*camera.horizontal.xyz+ uv.y*camera.vertical.xyz - camera.eye.xyz);
    vec3 pos = camera.eye.xyz;

    // Initial voxel coordinates
    ivec3 voxel_pos = ivec3(floor(pos));

    // Step direction
    ivec3 step = ivec3(sign(dir.x), sign(dir.y), sign(dir.z));

    // Distance to the next voxel boundary
    vec3 tMax = vec3(
        (step.x > 0 ? voxel_pos.x + 1.0 : voxel_pos.x) - pos.x,
        (step.y > 0 ? voxel_pos.y + 1.0 : voxel_pos.y) - pos.y,
        (step.z > 0 ? voxel_pos.z + 1.0 : voxel_pos.z) - pos.z
    ) / dir;

    // Distance to cross a voxel
    vec3 tDelta = vec3(
        step.x / dir.x,
        step.y / dir.y,
        step.z / dir.z
    );

    // Maximum iterations to prevent infinite loops
    int maxSteps = 512;

    for (int i = 0; i < maxSteps; i++) {
        // Check the current voxel
        uint voxel = get_voxel(vec3(voxel_pos));
        if (voxel != 0) {
            vec3 color = colors[voxel];
            imageStore(img_output, pix, vec4(color.xyz, 1.0)); // Store color
            float t = min(min(tMax.x, tMax.y), tMax.z); // Calculate the distance traveled
            imageStore(depth_output, pix, vec4(t)); // Store depth
            return;
        }

        // Move to the next voxel
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                voxel_pos.x += step.x;
                tMax.x += tDelta.x;
            } else {
                voxel_pos.z += step.z;
                tMax.z += tDelta.z;
            }
        } else {
            if (tMax.y < tMax.z) {
                voxel_pos.y += step.y;
                tMax.y += tDelta.y;
            } else {
                voxel_pos.z += step.z;
                tMax.z += tDelta.z;
            }
        }
    }

    imageStore(img_output, pix, vec4(0.,0.,0., 1.0)); // Black color for miss
    imageStore(depth_output, pix, vec4(1.0)); // Max depth for miss
}
