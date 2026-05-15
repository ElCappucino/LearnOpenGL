#version 330 core

out vec4 FragColor;

in vec3 WorldPos;
in vec3 Normal;
in vec2 TexCoords;

uniform vec3 camPos;

uniform sampler2D albedoMap;

uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

uniform float metallic;
uniform float roughness;
uniform float ao;

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0)
        * pow(1.0 - cosTheta, 5.0);
}

void main()
{
    vec3 albedo = pow(texture(albedoMap, TexCoords).rgb, vec3(2.2));

    vec3 N = normalize(Normal);

    vec3 V = normalize(camPos - WorldPos);

    vec3 R = reflect(-V, N);

    // base reflectivity
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // Fresnel
    vec3 F = fresnelSchlick(max(dot(N, V), 0.0), F0);

    // diffuse energy conservation
    vec3 kD = 1.0 - F;
    kD *= 1.0 - metallic;

    // diffuse irradiance
    vec3 irradiance = texture(irradianceMap, N).rgb;

    vec3 diffuse = irradiance * albedo;

    // specular IBL
    const float MAX_REFLECTION_LOD = 4.0;

    vec3 prefilteredColor =
        textureLod(
            prefilterMap,
            R,
            roughness * MAX_REFLECTION_LOD
        ).rgb;

    vec2 envBRDF =
        texture(
            brdfLUT,
            vec2(max(dot(N, V), 0.0), roughness)
        ).rg;

    vec3 specular =
        prefilteredColor *
        (F * envBRDF.x + envBRDF.y);

    vec3 ambient =
        (kD * diffuse + specular) * ao;

    // HDR tonemap
    vec3 color = ambient;

    color = color / (color + vec3(1.0));

    // gamma correct
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
}