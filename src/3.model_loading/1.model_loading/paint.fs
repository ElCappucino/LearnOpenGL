#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D fluidTexture;

void main() {
    float density = texture(fluidTexture, TexCoords).r;
    
    // Debug: If you see solid red, the quad is rendering but texture is empty
    // FragColor = vec4(1.0, 0.0, 0.0, 1.0); 

    // Real output: Blue ink that gets darker/more opaque with density
    if(density < 0.001) discard; 
    
    vec3 inkColor = vec3(0.0, 0.4, 0.9);
    FragColor = vec4(inkColor, clamp(density, 0.0, 1.0));
}