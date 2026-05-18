#version 330 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 norm;
layout(location = 2) in vec2 tex;
layout(location = 3) in vec3 tangent;
layout(location = 4) in vec3 bitangent;
layout(location = 5) in ivec4 boneIds; 
layout(location = 6) in vec4 weights;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

const int MAX_BONES = 100;
const int MAX_BONE_INFLUENCE = 4;
uniform mat4 finalBonesMatrices[MAX_BONES];

out vec2 TexCoords;
out vec3 FragPos;
out vec3 Normal;
out vec3 Tangent;
out vec3 Bitangent;

void main()
{
    vec4 totalPosition = vec4(0.0f);
    vec3 totalNormal = vec3(0.0f);
    vec3 totalTangent = vec3(0.0f);
    vec3 totalBitangent = vec3(0.0f);
    float totalWeight = 0.0f;

    for(int i = 0; i < MAX_BONE_INFLUENCE; i++)
    {
        // FIX 1: Change break to continue so we still evaluate structural sums completely
        if(boneIds[i] == -1) 
            continue;
            
        if(boneIds[i] >= MAX_BONES) 
        {
            continue;
        }
        
        // Accumulate transformation calculations
        vec4 localPosition = finalBonesMatrices[boneIds[i]] * vec4(pos, 1.0f);
        totalPosition += localPosition * weights[i];
        
        totalNormal    += mat3(finalBonesMatrices[boneIds[i]]) * norm * weights[i];
        totalTangent   += mat3(finalBonesMatrices[boneIds[i]]) * tangent * weights[i];
        totalBitangent += mat3(finalBonesMatrices[boneIds[i]]) * bitangent * weights[i];
        
        totalWeight += weights[i];
    }
    
    // FALLBACK: If this vertex has absolutely no valid skinning weights, map directly to base pose
    if (totalWeight < 0.01f)
    {
        totalPosition = vec4(pos, 1.0f);
        totalNormal    = norm;
        totalTangent   = tangent;
        totalBitangent = bitangent;
    }
    else
    {
        // FIX 2: Normalize by total weight to handle non-normalized export skin weights smoothly
        totalPosition /= totalWeight;
    }
    
    // Calculate world space position for lighting calculations
    FragPos = vec3(model * totalPosition);
    
    // Cleanly transform skin-deformed vectors to world space
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    Normal    = normalize(normalMatrix * totalNormal);
    Tangent   = normalize(normalMatrix * totalTangent);
    Bitangent = normalize(normalMatrix * totalBitangent);

    TexCoords = tex;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}