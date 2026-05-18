#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;

void main() {
    TexCoords = aTexCoords;
    
    // Copy position to modify it
    vec3 modifiedPos = aPos;
    
    // Only sway the top parts of the fire! 
    // aTexCoords.y goes from 0.0 (bottom) to 1.0 (top).
    float heightFactor = aTexCoords.y; 
    
    // Create a wave distortion over time
    float sway = sin(time * 3.0 + aPos.y * 2.0) * 0.15 * heightFactor;
    
    // Displace horizontally
    modifiedPos.x += sway; 
    
    gl_Position = projection * view * model * vec4(modifiedPos, 1.0);
}