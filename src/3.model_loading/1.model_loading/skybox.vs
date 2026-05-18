#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 WorldPos;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    WorldPos = aPos;
    // Strip translation from the view matrix so skybox stays centered on camera
    mat4 rotView = mat4(mat3(view)); 
    vec4 pos = projection * rotView * vec4(aPos, 1.0);
    
    // Trick the depth test by setting z to w (renders at farthest depth plane)
    gl_Position = pos.xyww;
}