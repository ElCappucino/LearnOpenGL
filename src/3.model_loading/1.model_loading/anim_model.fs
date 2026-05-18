#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;
in vec3 Tangent;
in vec3 Bitangent;

// Material Textures
uniform sampler2D texture_diffuse1;

// IBL / Skybox Maps
uniform samplerCube environmentMap; 

uniform vec3 cameraPos;
uniform vec3 lightPos; 
uniform vec3 lightColor;

void main()
{        
    // 1. Sample diffuse map cleanly
    vec3 albedo = texture(texture_diffuse1, TexCoords).rgb;
    
    vec3 N = normalize(Normal); 
    vec3 V = normalize(cameraPos - FragPos);

    // 2. Material Properties Constants (Simulating non-metallic character gear)
    float metallic  = 0.0;  // 0.0 = completely matte/cloth/skin, 1.0 = pure metal
    float roughness = 0.6;  // Higher values (0.5 - 0.8) blur reflections for rough surfaces
    
    // Base reflection for non-metals is roughly 4% (0.04)
    vec3 F0 = vec3(0.04); 
    // If it were a metal, the specular reflection would be tinted by the base albedo
    F0 = mix(F0, albedo, metallic);

    // 3. Ambient IBL Component (Fresnel reflection approximation)
    vec3 R = reflect(-V, N); 
    
    // Calculate how shiny the angle is (Fresnel effect: shiny at grazing angles)
    float cosTheta = max(dot(N, V), 0.0);
    vec3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    
    // kS is the reflection fraction, kD is the remaining diffuse fraction
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= (1.0 - metallic); // Metals have no diffuse component

    // Sample environment cubemap utilizing roughness to choose the blur layer
    // Multiplying roughness by the max mip-level splits sharp vs blurry reflections nicely
    vec3 prefilteredColor = textureLod(environmentMap, R, roughness * 6.0).rgb;
    
    // Combine diffuse ambient and specular ambient cleanly
    vec3 ambientDiffuse  = albedo * prefilteredColor; 
    vec3 ambientSpecular = prefilteredColor * F;
    vec3 ambient = (kD * ambientDiffuse + ambientSpecular) * 0.4; // 0.4 ambient intensity multiplier

    // 4. Direct Lighting Calculation
    vec3 L = normalize(lightPos - FragPos);
    
    // Diffuse component
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * albedo * lightColor;

    // Combine lighting passes
    vec3 color = ambient + diffuse;

    // 5. HDR Tone Mapping & Gamma Correction
    color = clamp(color, 0.0, 10.0);
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 1.5)); 

    FragColor = vec4(color, 1.0);
}