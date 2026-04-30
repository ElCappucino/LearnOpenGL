#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

#include <iostream>
#include <set>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 force;
    float mass = 4.0f;
};

struct Spring {
    int p1, p2;     // Indices of connected particles
    float restLen;
    float stiffness = 300.0f;
    float damping = 2.0f;
};

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 20.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, -90.0f);
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

const int CANVAS_RES = 200;
const float FLOOR_SIZE = 10.0f;
bool canvas[CANVAS_RES][CANVAS_RES] = { false };

// Buffer for rendering the points
unsigned int paintVAO, paintVBO;
std::vector<glm::vec3> paintPositions;

std::vector<Particle> particles;
std::vector<Spring> springs;

glm::vec3 getMouseWorldPos(GLFWwindow* window, glm::mat4 projection, glm::mat4 view) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Normalize coordinates to [-1, 1]
    float x = (2.0f * (float)xpos) / SCR_WIDTH - 1.0f;
    float y = 1.0f - (2.0f * (float)ypos) / SCR_HEIGHT;

    glm::vec4 ray_clip = glm::vec4(x, y, -1.0, 1.0);
    glm::vec4 ray_eye = glm::inverse(projection) * ray_clip;
    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0, 0.0);

    glm::vec3 ray_world = glm::vec3(glm::inverse(view) * ray_eye);
    ray_world = glm::normalize(ray_world);

    // Ray-Plane Intersection (Y = 0)
    // t = (plane_origin - ray_origin).y / ray_direction.y
    float t = -camera.Position.y / ray_world.y;
    return camera.Position + t * ray_world;
}

void initTetrahedralBrush(float scale, glm::vec3 offset) {
    particles.clear();
    springs.clear();

    // Raw mesh data from PaintBrush.obj
    std::vector<glm::vec3> meshPositions = {
        {-0.058779, -0.089477, -4.023622}, { -0.058779, 7.643711, -2.476540},
        { 3.425780, -0.089477, -2.011811}, {  2.085968, 7.643711, -1.238270},
        { 3.425780, -0.089477,  2.011811}, {  2.085968, 7.643711,  1.238270},
        {-0.058779, -0.089477,  4.023622}, { -0.058779, 7.643711,  2.476540},
        {-3.543338, -0.089477,  2.011811}, { -2.203526, 7.643711,  1.238270},
        {-3.543338, -0.089477, -2.011811}, { -2.203526, 7.643711, -1.238270},
        {-0.058779,  4.248201, -3.395509}, {  2.881818, 4.248201, -1.697755},
        { 2.881818,  4.248201,  1.697755}, { -0.058779, 4.248201,  3.395509},
        {-2.999376,  4.248201,  1.697755}, { -2.999376, 4.248201, -1.697755}
    };

    for (const auto& pos : meshPositions) {

       

        Particle p;
        // Apply Scale first, then Translation (Position)[cite: 4]
        glm::vec3 pLocal = pos;

        float minY = -0.089477f;
        float maxY = 7.643711f;
        float t = (pLocal.y - minY) / (maxY - minY);

        // Taper factor (adjust these two values)
        float bottomScale = 0.3f;   // how narrow the tip is
        float topScale = 1.0f;   // keep top unchanged

        float radiusScale = glm::mix(bottomScale, topScale, t);

        // Apply scaling ONLY on XZ (radius), not Y
        pLocal.x *= radiusScale;
        pLocal.z *= radiusScale;

        p.position = (pLocal * scale) + offset;
        p.velocity = glm::vec3(0.0f);
        p.force = glm::vec3(0.0f);
        p.mass = glm::mix(10.0f, 1.0f, t);
        particles.push_back(p);
    }

    std::set<std::pair<int, int>> springSet;

    auto addSpring = [&](int i, int j) {
        if (i > j) std::swap(i, j);
        if (springSet.count({ i,j })) return;

        springSet.insert({ i,j });
        float dist = glm::distance(particles[i].position, particles[j].position);
        springs.push_back({ i, j, dist, 1500.0f, 60.0f });
    };

    // 3. Connectivity from OBJ Faces
    // Note: OBJ indices start at 1, so we subtract 1
    std::vector<int> meshIndices = {
        6,4,18, 14,13,18, 13,2,18, 17,7,18, 16,7,18, 14,5,18, 4,10,6,
        17,10,18, 10,8,18, 10,6,18, 15,5,18, 11,1,14, 12,10,18, 9,17,11,
        7,18,11, 11,3,14, 15,14,18, 17,11,18, 14,4,18, 8,18,16, 15,18,16,
        17,8,18, 17,16,18, 1,18,14, 6,2,4, 14,4,13, 13,4,2, 17,16,7,
        16,5,7, 14,11,5, 4,2,10, 17,12,10, 10,6,8, 10,2,6, 15,14,5,
        11,3,1, 12,2,10, 9,7,17, 7,5,18, 11,5,3, 15,6,14, 17,7,11,
        14,6,4, 8,6,18, 16,6,15, 17,10,8, 17,8,16, 1,11,18, 13,1,18
    };

    for (size_t i = 0; i < meshIndices.size(); i += 3) {
        addSpring(meshIndices[i] - 1, meshIndices[i + 1] - 1);
        addSpring(meshIndices[i + 1] - 1, meshIndices[i + 2] - 1);
        addSpring(meshIndices[i + 2] - 1, meshIndices[i] - 1);
    }

    // 4. Internal "Volume" Spring - Set to TOP instead of average center
    float maxY = -999.0f;
    for (auto& p : particles) {
        if (p.position.y > maxY) maxY = p.position.y;
    }

    // Position the center particle at the very top of the brush mesh
    // We use offset.x and offset.z so it's centered horizontally
    glm::vec3 topCenter(offset.x, maxY, offset.z);

    int centerIdx = particles.size(); // This index will be the very last particle
    particles.push_back({ topCenter, glm::vec3(0), glm::vec3(0), 1.0f });

    // Connect every vertex to this top point to create the "handle" structure
    for (int i = 0; i < centerIdx; ++i) {
        // Increase stiffness significantly for the handle-to-tip connection
        float handleStiffness = 5000.0f;
        float handleDamping = 60.0f;
        float dist = glm::distance(particles[i].position, particles[centerIdx].position);
        springs.push_back({ i, centerIdx, dist, handleStiffness, handleDamping });
    }
}

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

    // 1. Initialize physics data
    initTetrahedralBrush(0.5f, glm::vec3(0.0f, 10.0f, 0.0f));

    // 2. Setup VAO/VBO for lines
    unsigned int lineVAO, lineVBO;
    glGenVertexArrays(1, &lineVAO);
    glGenBuffers(1, &lineVBO);

    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    // Reserve space for all spring vertices (2 vertices per spring, 3 floats per vertex)
    glBufferData(GL_ARRAY_BUFFER, springs.size() * 2 * 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // 5. Draw a simple floor cross for reference
    std::vector<float> floorVertices = {
        // First Triangle
        -10.0f, 0.0f, -10.0f,
         10.0f, 0.0f, -10.0f,
         10.0f, 0.0f,  10.0f,
         // Second Triangle
         10.0f, 0.0f,  10.0f,
         -10.0f, 0.0f,  10.0f,
         -10.0f, 0.0f, -10.0f
    };
    unsigned int floorVAO, floorVBO;
    glGenVertexArrays(1, &floorVAO);
    glGenBuffers(1, &floorVBO);
    glBindVertexArray(floorVAO);
    glBindBuffer(GL_ARRAY_BUFFER, floorVBO);
    glBufferData(GL_ARRAY_BUFFER, floorVertices.size() * sizeof(float), floorVertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glGenVertexArrays(1, &paintVAO);
    glGenBuffers(1, &paintVBO);
    glBindVertexArray(paintVAO);
    glBindBuffer(GL_ARRAY_BUFFER, paintVBO);
    // We'll update this dynamically, so start with NULL or a small size
    glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // draw in wireframe
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- PHYSICS UPDATE ---
        float substeps = 8;
        float subDelta = (deltaTime > 0.1f ? 0.1f : deltaTime) / substeps; // Cap dt for stability

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        glm::vec3 mouseWorld = getMouseWorldPos(window, projection, view);

        float hoverHeight = 1.0f; // Height above plane
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            hoverHeight = 0.0f; // Press down
        }

        // 3. Update the handle (the volume center particle)
        int handleIdx = particles.size() - 1;
        particles[handleIdx].position = mouseWorld + glm::vec3(0, hoverHeight, 0);
        particles[handleIdx].velocity = glm::vec3(0);

        for (int s = 0; s < substeps; ++s) {
            // 1. Force Accumulation
            for (int i = 0; i < particles.size(); ++i) {
                if (i == handleIdx) continue;
                particles[i].force = glm::vec3(0.0f, -9.81f * particles[i].mass, 0.0f);
            }

            for (auto& sp : springs) {
                glm::vec3 diff = particles[sp.p1].position - particles[sp.p2].position;
                float dist = glm::length(diff);
                if (dist > 0.0001f) {
                    glm::vec3 dir = diff / dist;
                    float springF = -sp.stiffness * (dist - sp.restLen);
                    glm::vec3 relVel = particles[sp.p1].velocity - particles[sp.p2].velocity;
                    float dampF = glm::dot(relVel, dir) * sp.damping;
                    particles[sp.p1].force += (springF - dampF) * dir;
                    particles[sp.p2].force -= (springF - dampF) * dir;
                }
            }

            // 2. Integration & Painting
            for (int i = 0; i < particles.size(); ++i) {
                if (i == handleIdx) continue;

                particles[i].velocity += (particles[i].force / particles[i].mass) * subDelta;
                particles[i].velocity *= 0.985f; // Global damping to stop spinning
                particles[i].position += particles[i].velocity * subDelta;

                // Floor Collision & Painting
                if (particles[i].position.y < 0.05f) { // Slightly above 0 for better detection
                    particles[i].position.y = 0.05f;
                    particles[i].velocity.y *= -0.1f;

                    // Map World XZ [-10, 10] to Canvas [0, RES]
                    int cx = (int)((particles[i].position.x + FLOOR_SIZE) / (2.0f * FLOOR_SIZE) * CANVAS_RES);
                    int cz = (int)((particles[i].position.z + FLOOR_SIZE) / (2.0f * FLOOR_SIZE) * CANVAS_RES);

                    if (cx >= 0 && cx < CANVAS_RES && cz >= 0 && cz < CANVAS_RES) {
                        if (!canvas[cx][cz]) { // Only add if not already painted
                            canvas[cx][cz] = true;
                            // Offset Y slightly to avoid Z-fighting with floor
                            paintPositions.push_back(glm::vec3(particles[i].position.x, 0.01f, particles[i].position.z));
                        }
                    }
                }
            }
        }

        // --- PREPARE DATA ---
        std::vector<float> lineVertices;
        for (const auto& sp : springs) {
            lineVertices.push_back(particles[sp.p1].position.x);
            lineVertices.push_back(particles[sp.p1].position.y);
            lineVertices.push_back(particles[sp.p1].position.z);
            lineVertices.push_back(particles[sp.p2].position.x);
            lineVertices.push_back(particles[sp.p2].position.y);
            lineVertices.push_back(particles[sp.p2].position.z);
        }

        // --- RENDER ---
        ourShader.use();
        

        

        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);
        ourShader.setMat4("model", glm::mat4(1.0f));

        // Draw Floor
        ourShader.setVec3("objectColor", glm::vec3(0.5f, 0.5f, 0.5f));
        glBindVertexArray(floorVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        if (!paintPositions.empty()) {
            ourShader.setVec3("objectColor", glm::vec3(0.0f, 0.0f, 0.0f)); // Black ink
            glBindVertexArray(paintVAO);
            glBindBuffer(GL_ARRAY_BUFFER, paintVBO);
            // Upload the current list of paint points
            glBufferData(GL_ARRAY_BUFFER, paintPositions.size() * sizeof(glm::vec3), paintPositions.data(), GL_DYNAMIC_DRAW);

            // Set point size (you may need to enable GL_PROGRAM_POINT_SIZE)
            glPointSize(5.0f);
            glDrawArrays(GL_POINTS, 0, (GLsizei)paintPositions.size());
        }


        // Draw Cylinder (Update buffer then draw)
        ourShader.setVec3("objectColor", glm::vec3(1.0f, 1.0f, 1.0f));
        glBindVertexArray(lineVAO);
        glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, lineVertices.size() * sizeof(float), lineVertices.data());
        glDrawArrays(GL_LINES, 0, (GLsizei)(lineVertices.size() / 3));

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
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
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
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    //float xpos = static_cast<float>(xposIn);
    //float ypos = static_cast<float>(yposIn);

    //if (firstMouse)
    //{
    //    lastX = xpos;
    //    lastY = ypos;
    //    firstMouse = false;
    //}

    //float xoffset = xpos - lastX;
    //float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    //lastX = xpos;
    //lastY = ypos;

    //camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
