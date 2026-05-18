#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <learnopengl/animator.h>
#include <learnopengl/model_animation.h>
#include <glm/gtx/string_cast.hpp>

#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
unsigned int loadHDRTexture(const char* path);
void renderCube();
void renderQuad();

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// movement
glm::vec3 charPosition = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 charFront = glm::vec3(0.0f, 0.0f, -1.0f); // initial forward
glm::vec3 charFrontTarget = glm::vec3(0.0f, 0.0f, -1.0f); // initial forward

bool hasMovementInput = false;
float charSpeed = 2.5f;
bool isMoving = false;
bool lastMouseDown = false;

bool movementLocked = false;

enum AnimState {
    IDLE = 1,
    IDLE_SLASH,
    SLASHING,
    SLASH_IDLE,
    IDLE_RUN,
    RUN_IDLE,
    RUN
};
enum AnimState charState = IDLE;
// camera
float cameraRadius = 10.0f;          // distance from model
float orbitYaw = 0.0f;        // horizontal angle (degrees)
float orbitPitch = 20.0f;     // vertical angle (degrees)
float smoothSpeed = 8.0f;  // higher = faster interpolation
float targetYaw = orbitYaw;
float targetPitch = orbitPitch;

unsigned int cubeVAO = 0;
unsigned int cubeVBO = 0;

unsigned int quadVAO = 0;
unsigned int quadVBO = 0;

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
    stbi_set_flip_vertically_on_load(true);

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile shaders
    // -------------------------
    Shader ourShader("1.model_loading.vs", "1.model_loading.fs");
    Shader equirectangularToCubemapShader("cubemap.vs", "equirectangular_to_cubemap.fs");
    Shader skyboxShader("skybox.vs", "skybox.fs");
    Shader animShader("anim_model.vs", "anim_model.fs");
    Shader fireShader("fire.vs", "fire.fs");

    // load models
    // -----------
    Model ourModel(FileSystem::getPath("resources/objects/Pillar/pillar.obj"));

    Model playerModel(FileSystem::getPath("resources/objects/Skeletal/playerModel/playerTPose.dae"));
    Animation idleAnimation(FileSystem::getPath("resources/objects/Skeletal/playerIdle/playerIdle.dae"), &playerModel);
    Animation runAnimation(FileSystem::getPath("resources/objects/Skeletal/playerRun/playerRun.dae"), &playerModel);
    Animation slashAnimation(FileSystem::getPath("resources/objects/Skeletal/playerSlash/playerSlash.dae"), &playerModel);

    Animator animator(&idleAnimation);
    
    float blendAmount = 0.0f;
    float blendRate = 0.055f;

    // skybox
    unsigned int hdrTexture = loadHDRTexture("resources/textures/hdr/the_sky_is_on_fire_4k.hdr");
    
    // draw in wireframe
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // 3. Set up the target Framebuffer and Renderbuffer for offline rendering
    unsigned int captureFBO, captureRBO;
    glGenFramebuffers(1, &captureFBO);
    glGenRenderbuffers(1, &captureRBO);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512); // Cubemap Resolution
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);

    unsigned int envCubemap;
    glGenTextures(1, &envCubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    for (unsigned int i = 0; i < 6; ++i)
    {
        // Initialize 6 empty floating-point color buffers (512x512)
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F,
            512, 512, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    // CRITICAL FIX: Change GL_LINEAR to GL_LINEAR_MIPMAP_LINEAR so textureLod can sample blurred levels!
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 5. Build 6 camera matrices pointing to each side of the cube
    glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    glm::mat4 captureViews[] =
    {
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    };

    // 6. RENDER THE EQUIRECTANGULAR MAP ONTO THE CUBEMAP FACES
    equirectangularToCubemapShader.use();
    equirectangularToCubemapShader.setInt("equirectangularMap", 0);
    equirectangularToCubemapShader.setMat4("projection", captureProjection);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, hdrTexture);

    glViewport(0, 0, 512, 512); // Match cubemap resolution
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    for (unsigned int i = 0; i < 6; ++i)
    {
        equirectangularToCubemapShader.setMat4("view", captureViews[i]);
        // Attach current cubemap face directly to our Framebuffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, envCubemap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderCube(); // Draws standard cube geometry to generate face 'i'
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // CRITICAL FIX: Generate the mipmaps for the cubemap now that the faces are filled!
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    // Reset viewport to window dimensions for your real loop
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        printf("movementLocked = %d\n", movementLocked);
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // --- Locate this state machine block inside your main loop ---
        switch (charState) {
        case IDLE:
            if (isMoving) {
                animator.CrossFade(&runAnimation, 0.2f);
                charState = IDLE_RUN;
            }
            break;
        case IDLE_RUN:
            if (!animator.IsBlending()) {
                charState = RUN;
            }
            break;
        case RUN:
            if (!isMoving) {
                animator.CrossFade(&idleAnimation, 0.2f);
                charState = RUN_IDLE;
            }
            break;
        case RUN_IDLE:
            if (!animator.IsBlending()) {
                charState = IDLE;
            }
            break;

        case IDLE_SLASH:
            // FIX: Only trigger the crossfade once. Do not let input logic evaluate 
            // flags until animator successfully switches tracks.
            printf("IDLE_SLASH\n");
            animator.CrossFade(&slashAnimation, 0.2f); // Swapped to 0.2f for snappier entry
            charState = SLASHING;
            break;

        case SLASHING:
            // FIX: Enforce absolute isolation. The slash animation MUST complete 
            // its timeline loop before we poll any movement flags.
            if (animator.AnimationFinished())
            {
                if (isMoving)
                    animator.CrossFade(&runAnimation, 0.2f);
                else
                    animator.CrossFade(&idleAnimation, 0.2f);

                charState = SLASH_IDLE;
            }
            break;

        case SLASH_IDLE:
            if (!animator.IsBlending()) {
                charState = isMoving ? RUN : IDLE;
                movementLocked = false;
            }
            break;
        }
        //printf("State = %d\n", (int)charState);
        float t = 0.1f;
        charFront = glm::mix(charFront, charFrontTarget, t);

        animator.UpdateAnimation(deltaTime);

        // render setup
        // ------------
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Calculate Camera Uniform view & projection matrices once per frame
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

        float lerpFactor = 1.0f - expf(-smoothSpeed * deltaTime);
        orbitYaw = glm::mix(orbitYaw, targetYaw, lerpFactor);
        orbitPitch = glm::mix(orbitPitch, targetPitch, lerpFactor);

        float smoothYawRad = glm::radians(orbitYaw);
        float smoothPitchRad = glm::radians(orbitPitch);

        float camX = cameraRadius * cos(smoothPitchRad) * sin(smoothYawRad);
        float camY = cameraRadius * sin(smoothPitchRad);
        float camZ = cameraRadius * cos(smoothPitchRad) * cos(smoothYawRad);

        glm::vec3 cameraPos = charPosition + glm::vec3(camX, camY, camZ);
        glm::vec3 charPosWithOffset = charPosition + glm::vec3(0.0f, 1.0f, 0.0f);

        glm::mat4 view = glm::lookAt(cameraPos, charPosWithOffset, glm::vec3(0.0f, 1.0f, 0.0f));

        // ==========================================
        // 1. RENDER PLAYER MODEL (ANIMATED)
        // ==========================================
        animShader.use();
        animShader.setMat4("projection", projection);
        animShader.setMat4("view", view);

        animShader.setVec3("cameraPos", cameraPos);
        animShader.setVec3("lightPos", glm::vec3(2.0f, 4.0f, 2.0f));
        animShader.setVec3("lightColor", glm::vec3(0.01f, 0.01f, 0.01f));

        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
        animShader.setInt("environmentMap", 5);

        auto transforms = animator.GetFinalBoneMatrices();
        for (int i = 0; i < transforms.size(); ++i)
            animShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", transforms[i]);

        glm::mat4 model = glm::mat4(1.0f);
 
        model = glm::translate(model, charPosition);

        // --- FIXED MOVEMENT-ORIENTED FACING LOGIC ---
        // Rotate the character to look down their actual, interpolated movement vector
        if (glm::length(charFront) > 0.001f)
        {
            glm::vec3 lookDir = glm::normalize(glm::vec3(charFront.x, 0.0f, charFront.z));
            // Calculate orientation down the movement direction track
            float angle = atan2(lookDir.x, lookDir.z);
            model = glm::rotate(model, angle, glm::vec3(0.0f, 1.0f, 0.0f));
        }

        model = glm::scale(model, glm::vec3(0.1f));
        // ---------------------------------------------

        animShader.setMat4("model", model);
        playerModel.Draw(animShader);

        // ==========================================
        // 2. RENDER ENVIRONMENT PILLARS (STATIC)
        // ==========================================
        ourShader.use();
        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);
        ourShader.setVec3("cameraPos", cameraPos);

        ourShader.setVec3("lightPos", glm::vec3(2.0f, 4.0f, 2.0f));
        ourShader.setVec3("lightColor", glm::vec3(0.01f, 0.01f, 0.01f));
        ourShader.setFloat("roughnessModifier", 1.0f);

        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
        ourShader.setInt("environmentMap", 5);

        // Reset model matrix back to Identity since ourModel positions are relative to world origin
        model = glm::mat4(1.0f);
        ourShader.setMat4("model", model);
        ourModel.Draw(ourShader);
        //playerModel.Draw(ourShader);
        // ==========================================
        // 3. RENDER SKYBOX BACKGROUND (Once, at the end!)
        // ==========================================
        glDepthFunc(GL_LEQUAL);

        skyboxShader.use();
        skyboxShader.setMat4("view", view);
        skyboxShader.setMat4("projection", projection);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
        skyboxShader.setInt("skybox", 0);

        renderCube();

        glDepthFunc(GL_LESS); // Reset back to default standard depth testing

        // ==========================================
        // 4. RENDER PROCEDURAL FIRE (CROSSED X-BILLBOARD)
        // ==========================================
        glEnable(GL_BLEND);

        glBlendFunc(GL_ONE, GL_ONE);

        glDepthMask(GL_FALSE);

        fireShader.use();
        fireShader.setMat4("projection", projection);
        fireShader.setMat4("view", view);
        fireShader.setFloat("time", static_cast<float>(glfwGetTime()));

        glm::vec3 firePos = glm::vec3(5.5f, 0.9f, -1.0f); 
        glm::vec3 fireScale = glm::vec3(2.0f, 2.0f, 1.0f);

        // Define fixed angles for an intersecting 'X' pattern (0 and 90 degrees)
        // Tip: You can add a third angle (e.g., 45.0f) for an even denser "*" shape!
        float angles[] = { 0.0f, 45.0f, 90.0f, 135.0f };

        for (int i = 0; i < 4; ++i)
        {
            glm::mat4 modelCrossed = glm::mat4(1.0f);
            modelCrossed = glm::translate(modelCrossed, firePos);
            modelCrossed = glm::rotate(modelCrossed, glm::radians(angles[i]), glm::vec3(0.0f, 1.0f, 0.0f));
            modelCrossed = glm::scale(modelCrossed, fireScale);

            fireShader.setMat4("model", modelCrossed);

            // Send a unique phase offset variable to the shader for each plane
            fireShader.setFloat("planeOffset", static_cast<float>(i) * 15.5f);

            renderQuad();
        }
        glDepthMask(GL_TRUE);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_BLEND);

        // Swap buffers and poll
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    float smoothYawRad = glm::radians(orbitYaw);
    glm::vec3 camForwardXZ = glm::vec3(-sin(smoothYawRad), 0.0f, -cos(smoothYawRad));
    if (glm::length(camForwardXZ) > 0.0f)
        camForwardXZ = glm::normalize(camForwardXZ);

    glm::vec3 camRightXZ = glm::normalize(glm::cross(camForwardXZ, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 moveDir(0.0f);

    // 1. Handle Slash Input Trigger
    bool mouseDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (mouseDown && !lastMouseDown)
    {
        if (charState == IDLE || charState == RUN)
        {
            charState = IDLE_SLASH;
            movementLocked = true;
        }
    }
    lastMouseDown = mouseDown;

    // Check if player is currently locked in an attack animation
    //movementLocked = (charState == IDLE_SLASH || charState == SLASHING || charState == SLASH_IDLE);

    // 2. Gather Movement Input
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) moveDir += camForwardXZ;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) moveDir -= camForwardXZ;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) moveDir -= camRightXZ;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) moveDir += camRightXZ;

    // 3. Process Movement State Safely
    if (glm::length(moveDir) > 0.0f)
    {
        if (movementLocked)
        {
            // Force movement flags off completely while locked
            isMoving = false;
        }
        else
        {
            moveDir = glm::normalize(moveDir);
            charPosition += moveDir * charSpeed * deltaTime;
            charFrontTarget = moveDir;
            isMoving = true;
        }
    }
    else
    {
        isMoving = false;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    // Apply to target rotation only
    targetYaw -= xoffset;
    targetPitch -= yoffset;

    // Clamp pitch
    if (targetPitch > 89.0f) targetPitch = 89.0f;
    if (targetPitch < -89.0f) targetPitch = -89.0f;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

unsigned int loadHDRTexture(const char* path)
{
    stbi_set_flip_vertically_on_load(true);
    int width, height, nrComponents;
    float* data = stbi_loadf(path, &width, &height, &nrComponents, 0);
    unsigned int hdrTexture = 0;

    if (data)
    {
        glGenTextures(1, &hdrTexture);
        glBindTexture(GL_TEXTURE_2D, hdrTexture);

        // CRITICAL: Notice GL_RGB16F floating point internal texture format
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
        std::cout << "Successfully loaded HDR texture map!" << std::endl;
    }
    else
    {
        std::cout << "Failed to load HDR texture at path: " << path << std::endl;
    }

    return hdrTexture;
}

void renderCube()
{
    if (cubeVAO == 0)
    {
        float vertices[] = {
            // back face
            -1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
            1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
            // front face
            -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
             // left face
             -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
             -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
             // right face
              1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
              1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
              // bottom face
              -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
               1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
               // top face
               -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
                1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f
        };
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindVertexArray(cubeVAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}

void renderQuad() {
    if (quadVAO == 0) {
        float quadVertices[] = {
            // Positions        // Texture Coords
            -0.5f,  0.5f, 0.0f,  0.0f, 1.0f,
            -0.5f, -0.5f, 0.0f,  0.0f, 0.0f,
             0.5f,  0.5f, 0.0f,  1.0f, 1.0f,
             0.5f, -0.5f, 0.0f,  1.0f, 0.0f,
        };
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}
