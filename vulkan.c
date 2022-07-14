/*
 * Copyright © 2019 nyorain
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Author: nyorain <nyorain@gmail.com>
 */


#include "kms-quads.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <errno.h>

#include <vulkan.frag.h>
#include <vulkan.vert.h>

// This corresponds to the XRGB drm format.
// The egl format hardcodes this format so we can probably too.
// It's guaranteed to be supported by the vulkan spec for everything
// we need. SRGB is the correct choice here, as always. You'd see
// that when rendering a texture.
static const VkFormat format = VK_FORMAT_B8G8R8A8_SRGB;

struct vk_device {
    VkInstance instance;
    VkDebugUtilsMessengerEXT messenger;

    // whether the required extensions for explicit fencing are supported
    bool explicit_fencing;

    struct {
        PFN_vkCreateDebugUtilsMessengerEXT createDebugUtilsMessengerEXT;
        PFN_vkDestroyDebugUtilsMessengerEXT destroyDebugUtilsMessengerEXT;
        PFN_vkGetMemoryFdPropertiesKHR getMemoryFdPropertiesKHR;
        PFN_vkGetSemaphoreFdKHR getSemaphoreFdKHR;
        PFN_vkImportSemaphoreFdKHR importSemaphoreFdKHR;
    } api;

    VkPhysicalDevice phdev;
    VkDevice dev;

    uint32_t queue_family;
    VkQueue queue;

    // pipeline
    VkDescriptorSetLayout ds_layout;
    VkRenderPass rp;
    VkPipelineLayout pipe_layout;
    VkPipeline pipe;
    VkCommandPool command_pool;
    VkDescriptorPool ds_pool;
};

struct vk_image {
    struct buffer buffer;

    VkDeviceMemory memories[4]; // worst case: 4 planes, 4 memory objects
    VkImage image;
    VkImageView image_view;
    VkCommandBuffer cb;
    VkFramebuffer fb;
    bool first;

    VkBuffer ubo;
    VkDeviceMemory ubo_mem;
    void *ubo_map;
    VkDescriptorSet ds;

    // We have to use a semaphore here since we want to "wait for it
    // on the device" (i.e. only start rendering when the semaphore
    // is signaled) and that isn't possible with a fence.
    VkSemaphore buffer_semaphore; // signaled by kernal when image can be reused

    // vulkan can signal a semaphore and a fence when a command buffer
    // has completed, so we can use either here without any significant
    // difference (the exporting semantics are the same for both).
    VkSemaphore render_semaphore; // signaled by vulkan when rendering finishes

    // We don't need this theoretically. But the validation layers
    // are happy if we signal them via this fence that execution
    // has finished.
    VkFence render_fence; // signaled by vulkan when rendering finishes
};

// #define vk_error(res, fmt, ...)
#define vk_error(res, fmt) error(fmt ": %s (%d)\n", vulkan_strerror(res), res)

// Returns a VkResult value as string.
static const char *vulkan_strerror(VkResult err) {
    #define ERR_STR(r) case VK_ ##r: return #r

    switch (err) {
        ERR_STR(SUCCESS);
        ERR_STR(NOT_READY);
        ERR_STR(TIMEOUT);
        ERR_STR(EVENT_SET);
        ERR_STR(EVENT_RESET);
        ERR_STR(INCOMPLETE);
        ERR_STR(ERROR_OUT_OF_HOST_MEMORY);
        ERR_STR(ERROR_OUT_OF_DEVICE_MEMORY);
        ERR_STR(ERROR_INITIALIZATION_FAILED);
        ERR_STR(ERROR_DEVICE_LOST);
        ERR_STR(ERROR_MEMORY_MAP_FAILED);
        ERR_STR(ERROR_LAYER_NOT_PRESENT);
        ERR_STR(ERROR_EXTENSION_NOT_PRESENT);
        ERR_STR(ERROR_FEATURE_NOT_PRESENT);
        ERR_STR(ERROR_INCOMPATIBLE_DRIVER);
        ERR_STR(ERROR_TOO_MANY_OBJECTS);
        ERR_STR(ERROR_FORMAT_NOT_SUPPORTED);
        ERR_STR(ERROR_FRAGMENTED_POOL);
        ERR_STR(ERROR_UNKNOWN);
        ERR_STR(ERROR_OUT_OF_POOL_MEMORY);
        ERR_STR(ERROR_INVALID_EXTERNAL_HANDLE);
        ERR_STR(ERROR_FRAGMENTATION);
        ERR_STR(ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
        ERR_STR(ERROR_SURFACE_LOST_KHR);
        ERR_STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        ERR_STR(SUBOPTIMAL_KHR);
        ERR_STR(ERROR_OUT_OF_DATE_KHR);
        ERR_STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        ERR_STR(ERROR_VALIDATION_FAILED_EXT);
        ERR_STR(ERROR_INVALID_SHADER_NV);
        ERR_STR(ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
        ERR_STR(ERROR_NOT_PERMITTED_EXT);
        ERR_STR(ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
        ERR_STR(THREAD_IDLE_KHR);
        ERR_STR(THREAD_DONE_KHR);
        ERR_STR(OPERATION_DEFERRED_KHR);
        ERR_STR(OPERATION_NOT_DEFERRED_KHR);
        ERR_STR(PIPELINE_COMPILE_REQUIRED_EXT);
        default:
            return "<unknown>";
    }

    #undef ERR_STR
}

static VkImageAspectFlagBits mem_plane_ascpect(unsigned i) {
    switch(i) {
        case 0: return VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT;
        case 1: return VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT;
        case 2: return VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT;
        case 3: return VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT;
        default: assert(false); // unreachable
    }
}

int find_mem_type(VkPhysicalDevice phdev, VkMemoryPropertyFlags flags, uint32_t req_bits) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(phdev, &props);

    for (unsigned i = 0u; i < props.memoryTypeCount; ++i) {
        if (req_bits & (1 << i)) {
            if ((props.memoryTypes[i].propertyFlags & flags) == flags) {
                return i;
            }
        }
    }

    return -1;
}

static bool has_extension(const VkExtensionProperties *avail, uint32_t availc, const char *req) {
    // check if all required extensions are supported
    for (size_t j = 0; j < availc; ++j) {
        if (!strcmp(avail[j].extensionName, req)) {
            return true;
        }
    }

    return false;
}

static VkBool32 debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT *debug_data,
    void *data
) {

    ((void) data);
    ((void) type);

    // we ignore some of the non-helpful warnings
    // static const char *const ignored[] = {};
    // if (debug_data->pMessageIdName) {
    // 	for (unsigned i = 0; i < sizeof(ignored) / sizeof(ignored[0]); ++i) {
    // 		if (!strcmp(debug_data->pMessageIdName, ignored[i])) {
    // 			return false;
    // 		}
    // 	}
    // }

    const char* importance = "UNKNOWN";
    switch(severity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            importance = "ERROR";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            importance = "WARNING";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            importance = "INFO";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            importance = "VERBOSE";
            break;
        default:
            break;
    }

    debug("%s: %s (%s, %d)\n", importance, debug_data->pMessage,
        debug_data->pMessageIdName, debug_data->messageIdNumber);
    if (debug_data->queueLabelCount > 0) {
        const char *name = debug_data->pQueueLabels[0].pLabelName;
        if (name) {
            debug("    last queue label '%s'\n", name);
        }
    }

    if (debug_data->cmdBufLabelCount > 0) {
        const char *name = debug_data->pCmdBufLabels[0].pLabelName;
        if (name) {
            debug("    last cmdbuf label '%s'\n", name);
        }
    }

    for (unsigned i = 0; i < debug_data->objectCount; ++i) {
        if (debug_data->pObjects[i].pObjectName) {
            debug("    involving '%s'\n", debug_data->pMessage);
        }
    }

    // Returning true not allowed by spec but helpful for debugging
    // makes function that caused the error return validation_failed
    // error which we can detect
    // return true;

    return false;
}

void vk_device_destroy(struct vk_device *device) {
    if (device->pipe) {
        vkDestroyPipeline(device->dev, device->pipe, NULL);
    }
    if (device->rp) {
        vkDestroyRenderPass(device->dev, device->rp, NULL);
    }
    if (device->pipe_layout) {
        vkDestroyPipelineLayout(device->dev, device->pipe_layout, NULL);
    }
    if (device->command_pool) {
        vkDestroyCommandPool(device->dev, device->command_pool, NULL);
    }
    if (device->ds_layout) {
        vkDestroyDescriptorSetLayout(device->dev, device->ds_layout, NULL);
    }
    if (device->ds_pool) {
        vkDestroyDescriptorPool(device->dev, device->ds_pool, NULL);
    }
    if (device->dev) {
        vkDestroyDevice(device->dev, NULL);
    }
    if (device->messenger && device->api.destroyDebugUtilsMessengerEXT) {
        device->api.destroyDebugUtilsMessengerEXT(device->instance,
            device->messenger, NULL);
    }
    if (device->instance) {
        vkDestroyInstance(device->instance, NULL);
    }
    free(device);
}

static bool init_pipeline(struct vk_device *dev) {
    VkResult res = vkCreateRenderPass(
        dev->dev,

        // render pass
        // We don't care about previous contents of the image since
        // we always render the full image. For incremental presentation you
        // have to use LOAD_OP_STORE and a valid image layout.
        &(VkRenderPassCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .attachmentCount = 1,
            .pAttachments = &(VkAttachmentDescription) {
                .flags = 0,
                .format = format,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                // attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                // can basically be anything since we have to manually transition
                // the image afterwards anyways (see depdency reasoning below)
                .finalLayout = VK_IMAGE_LAYOUT_GENERAL
            },
            .subpassCount = 1,
            .pSubpasses = &(VkSubpassDescription) {
                .flags = 0,
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .inputAttachmentCount = 0,
                .pInputAttachments = 0,
                .colorAttachmentCount = 1,
                .pColorAttachments = &(VkAttachmentReference) {
                    .attachment = 0u,
                    .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                },
                .pResolveAttachments = 0,
                .pDepthStencilAttachment = 0,
                .preserveAttachmentCount = 0,
                .pPreserveAttachments = 0,
            },
            // Note how we don't specify any (external) subpass dependencies.
            // The transfer of an image to an external queue (i.e. transfer logical
            // ownership of the image from the vulkan driver to drm) can't be represented
            // as a subpass dependency, so we have to transition the image
            // after and before a renderpass manually anyways.
            .dependencyCount = 0,
            .pDependencies = 0,
        },
        NULL,
        &dev->rp
    );
    if (res != VK_SUCCESS) {
        vk_error(res, "vkCreateRenderPass");
        return false;
    }

    // pipeline layout
    res = vkCreateDescriptorSetLayout(
        dev->dev,
        &(VkDescriptorSetLayoutCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = 0,
            .flags = 0,
            .bindingCount = 1u,
            .pBindings = &(VkDescriptorSetLayoutBinding) {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .pImmutableSamplers = 0,
            },
        },
        NULL,
        &dev->ds_layout
    );
    if (res != VK_SUCCESS) {
        vk_error(res, "vkCreateDescriptorSetLayout");
        return false;
    }

    res = vkCreatePipelineLayout(
        dev->dev,
        &(VkPipelineLayoutCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = 0,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &dev->ds_layout,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = 0,
        },
        NULL,
        &dev->pipe_layout
    );
    if (res != VK_SUCCESS) {
        vk_error(res, "vkCreatePipelineLayout");
        return false;
    }

    // pipeline
    VkShaderModule vert_module;
    VkShaderModule frag_module;

    VkShaderModuleCreateInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    si.codeSize = sizeof(vulkan_vert_data);
    si.pCode = vulkan_vert_data;
    res = vkCreateShaderModule(dev->dev, &si, NULL, &vert_module);
    if (res != VK_SUCCESS) {
        vk_error(res, "Failed to create vertex shader module");
        return false;
    }

    si.codeSize = sizeof(vulkan_frag_data);
    si.pCode = vulkan_frag_data;
    res = vkCreateShaderModule(dev->dev, &si, NULL, &frag_module);
    if (res != VK_SUCCESS) {
        vk_error(res, "Failed to create fragment shader module");
        vkDestroyShaderModule(dev->dev, vert_module, NULL);
        return false;
    }

    VkPipelineShaderStageCreateInfo pipe_stages[2] = {{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vert_module, "main", NULL
        }, {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag_module, "main", NULL
        }
    };

    // info
    VkPipelineInputAssemblyStateCreateInfo assembly = {0};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;

    VkPipelineRasterizationStateCreateInfo rasterization = {0};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.f;

    VkPipelineColorBlendAttachmentState blend_attachment = {0};
    blend_attachment.blendEnable = false;
    blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT;

    VkPipelineColorBlendStateCreateInfo blend = {0};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;

    VkPipelineMultisampleStateCreateInfo multisample = {0};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineViewportStateCreateInfo viewport = {0};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkDynamicState dynStates[2] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dynamic = {0};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.pDynamicStates = dynStates;
    dynamic.dynamicStateCount = 2;

    VkPipelineVertexInputStateCreateInfo vertex = {0};
    vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkGraphicsPipelineCreateInfo pipe_info = {0};
    pipe_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipe_info.layout = dev->pipe_layout;
    pipe_info.renderPass = dev->rp;
    pipe_info.subpass = 0;
    pipe_info.stageCount = 2;
    pipe_info.pStages = pipe_stages;

    pipe_info.pInputAssemblyState = &assembly;
    pipe_info.pRasterizationState = &rasterization;
    pipe_info.pColorBlendState = &blend;
    pipe_info.pMultisampleState = &multisample;
    pipe_info.pViewportState = &viewport;
    pipe_info.pDynamicState = &dynamic;
    pipe_info.pVertexInputState = &vertex;

    // could use a cache here for faster loading
    VkPipelineCache cache = VK_NULL_HANDLE;
    res = vkCreateGraphicsPipelines(dev->dev, cache, 1, &pipe_info, NULL, &dev->pipe);
    vkDestroyShaderModule(dev->dev, vert_module, NULL);
    vkDestroyShaderModule(dev->dev, frag_module, NULL);
    if (res != VK_SUCCESS) {
        error("failed to create vulkan pipeline: %d\n", res);
        return false;
    }

    return true;
}

struct vk_device *vk_device_create(struct device *device) {
    // check for drm device support
    // vulkan requires modifier support to import dma bufs
    if (!device->fb_modifiers) {
        debug("Can't use vulkan since drm doesn't support modifiers\n");
        return NULL;
    }

    // query extension support
    uint32_t avail_extc = 0;
    VkResult res;
    res = vkEnumerateInstanceExtensionProperties(NULL, &avail_extc, NULL);
    if ((res != VK_SUCCESS) || (avail_extc == 0)) {
        vk_error(res, "Could not enumerate instance extensions (1)");
        return NULL;
    }

    VkExtensionProperties *avail_exts = calloc(avail_extc, sizeof(*avail_exts));
    res = vkEnumerateInstanceExtensionProperties(NULL, &avail_extc, avail_exts);
    if (res != VK_SUCCESS) {
        free(avail_exts);
        vk_error(res, "Could not enumerate instance extensions (2)");
        return NULL;
    }

    for (size_t j = 0; j < avail_extc; ++j) {
        debug("Vulkan Instance extensions %s\n", avail_exts[j].extensionName);
    }

    struct vk_device *vk_dev = calloc(1, sizeof(*vk_dev));
    assert(vk_dev);

    // create instance
    const char *req = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    const char** enable_exts = NULL;
    uint32_t enable_extc = 0;
    if (!has_extension(avail_exts, avail_extc, req)) {
        error("extension " VK_EXT_DEBUG_UTILS_EXTENSION_NAME " is required");
        return NULL;
    }

    free(avail_exts);

    // layer reports error in api usage to debug callback
    const char *layers[] = {
        "VK_LAYER_KHRONOS_validation",
    };

    res = vkCreateInstance(
        &(VkInstanceCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .pApplicationInfo = &(VkApplicationInfo) {
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pNext = NULL,
                .pApplicationName = "kmscube_vulkan",
                .applicationVersion = 1,
                .pEngineName = "kmscube_vulkan",
                .engineVersion = 1,
                .apiVersion = VK_MAKE_VERSION(1,1,0),
            },
            .enabledLayerCount = 0 /* ARRAY_LENGTH(layers) */,
            .ppEnabledLayerNames = NULL /* layers */,
            .enabledExtensionCount = 1,
            .ppEnabledExtensionNames = (const char*[]) {
                VK_EXT_DEBUG_UTILS_EXTENSION_NAME
            },
        },
        NULL,
        &vk_dev->instance
    );
    if (res != VK_SUCCESS) {
        vk_error(res, "Could not create instance");
        goto error;
    }

    vk_dev->api.createDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(vk_dev->instance, "vkCreateDebugUtilsMessengerEXT");
    vk_dev->api.destroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(vk_dev->instance, "vkDestroyDebugUtilsMessengerEXT");

    if (!vk_dev->api.createDebugUtilsMessengerEXT || !vk_dev->api.destroyDebugUtilsMessengerEXT) {
        error("Could not resolve debugging utils vulkan procedures.\n");
        return NULL;
    }

    vk_dev->api.createDebugUtilsMessengerEXT(
        vk_dev->instance,
        &(VkDebugUtilsMessengerCreateInfoEXT) {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = NULL,
            .flags = 0,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debug_callback,
            .pUserData = 0,
        },
        NULL,
        &vk_dev->messenger
    );

    // enumerate physical devices to find the one matching the given
    // gbm device.
    uint32_t num_phdevs;
    res = vkEnumeratePhysicalDevices(vk_dev->instance, &num_phdevs, NULL);
    if (res != VK_SUCCESS || num_phdevs == 0) {
        vk_error(res, "Could not retrieve physical device");
        goto error;
    }

    VkPhysicalDevice *phdevs = calloc(num_phdevs, sizeof(*phdevs));
    res = vkEnumeratePhysicalDevices(vk_dev->instance, &num_phdevs, phdevs);
    if (res != VK_SUCCESS || num_phdevs == 0) {
        free(phdevs);
        vk_error(res, "Could not retrieve physical device");
        goto error;
    }

//	drmPciBusInfoPtr pci = drm_dev->businfo.pci;
//	debug("PCI bus: %04x:%02x:%02x.%x\n", pci->domain,
//		pci->bus, pci->dev, pci->func);

    VkExtensionProperties *phdev_exts = NULL;
    uint32_t phdev_extc = 0;
    VkPhysicalDevice phdev = VK_NULL_HANDLE;
    for (unsigned i = 0u; i < num_phdevs; ++i) {
        VkPhysicalDevice phdevi = phdevs[i];

        VkPhysicalDeviceProperties props = {0};
        vkGetPhysicalDeviceProperties(phdevi, &props);
        if (strcmp(props.deviceName, "V3D 4.2") == 0) {
            phdev = phdevi;
        } else {
            continue;
        }

        VkResult res;
        res = vkEnumerateDeviceExtensionProperties(phdev, NULL, &phdev_extc, NULL);
        if ((res != VK_SUCCESS) || (phdev_extc == 0)) {
            phdev_extc = 0;
            vk_error(res, "Could not enumerate device extensions (1)");
            return false;
        }

        phdev_exts = realloc(phdev_exts, sizeof(*phdev_exts) * phdev_extc);
        res = vkEnumerateDeviceExtensionProperties(phdev, NULL, &phdev_extc, phdev_exts);
        if (res != VK_SUCCESS) {
            vk_error(res, "Could not enumerate device extensions (2)");
            return false;
        }
    }

    free(phdevs);
    if (phdev == VK_NULL_HANDLE) {
        error("Can't find vulkan physical device for drm dev\n");
        goto error;
    }

    for (size_t j = 0; j < phdev_extc; ++j) {
        debug("Vulkan Device extensions %s\n", phdev_exts[j].extensionName);
    }

    vk_dev->phdev = phdev;

    // query extensions
    const char* dev_exts[8];
    uint32_t dev_extc = 0;

    const char* mem_exts[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME, // required by drm ext

        // NOTE: strictly speaking this extension is required to
        // correctly transfer image ownership but since no mesa
        // driver implements its yet (no even an updated patch for that),
        // let's see how far we get without it
        // VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
    };

    for (unsigned i = 0u; i < ARRAY_LENGTH(mem_exts); ++i) {
        if (!has_extension(phdev_exts, phdev_extc, mem_exts[i])) {
            error("Physical device doesn't supported required extension: %s\n", mem_exts[i]);
            return NULL;
        } else {
            dev_exts[dev_extc++] = mem_exts[i];
        }
    }

    // explicit fencing extensions
    // we currently only import/export semaphores
    vk_dev->explicit_fencing = true;
    const char* sync_exts[] = {
        // VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    };

    for (unsigned i = 0u; i < ARRAY_LENGTH(sync_exts); ++i) {
        if (!has_extension(phdev_exts, phdev_extc, sync_exts[i])) {
            error("Physical device doesn't supported extension %s, which "
                "is required for explicit fencing. Will disable explicit "
                "fencing but that is a suboptimal workaround",
                dev_exts[i]);
            return NULL;
        } else {
            dev_exts[dev_extc++] = sync_exts[i];
        }
    }

    // create device
    // queue families
    uint32_t qfam_count;
    vkGetPhysicalDeviceQueueFamilyProperties(phdev, &qfam_count, NULL);
    VkQueueFamilyProperties *qprops = calloc(sizeof(*qprops), qfam_count);
    vkGetPhysicalDeviceQueueFamilyProperties(phdev, &qfam_count, qprops);

    uint32_t qfam = 0xFFFFFFFFu; // graphics queue family
    for (unsigned i = 0u; i < qfam_count; ++i) {
        if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            qfam = i;
            break;
        }
    }

    // vulkan standard guarantees that the must be at least one graphics
    // queue family
    assert(qfam != 0xFFFFFFFFu);
    vk_dev->queue_family = qfam;

    // info
    float prio = 1.f;
    res = vkCreateDevice(
        phdev,
        &(VkDeviceCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &(VkDeviceQueueCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .pNext = NULL,
                .flags = 0,
                .queueFamilyIndex = qfam,
                .queueCount = 1,
                .pQueuePriorities = &prio,
            },
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = NULL,
            .enabledExtensionCount = dev_extc,
            .ppEnabledExtensionNames = dev_exts,
            .pEnabledFeatures = NULL,
        },
        NULL,
        &vk_dev->dev
    );
    if (res != VK_SUCCESS){
        vk_error(res, "Failed to create vulkan device");
        goto error;
    }

    vkGetDeviceQueue(vk_dev->dev, vk_dev->queue_family, 0, &vk_dev->queue);

    // command pool
    res = vkCreateCommandPool(
        vk_dev->dev,
        &(VkCommandPoolCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = NULL,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = vk_dev->queue_family,
        },
        NULL,
        &vk_dev->command_pool
    );
    if (res != VK_SUCCESS) {
        vk_error(res, "vkCreateCommandPool");
        goto error;
    }

    // descriptor pool
    res = vkCreateDescriptorPool(
        vk_dev->dev,
        &(VkDescriptorPoolCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .maxSets = BUFFER_QUEUE_DEPTH,
            .poolSizeCount = 1u,
            .pPoolSizes = &(VkDescriptorPoolSize) {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = BUFFER_QUEUE_DEPTH,
            },
        },
        NULL,
        &vk_dev->ds_pool
    );
    if (res != VK_SUCCESS) {
        vk_error(res, "vkCreateDescriptorPool");
        goto error;
    }

    VkExternalSemaphoreProperties props = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
        .pNext = NULL,
        .exportFromImportedHandleTypes = 0,
        .compatibleHandleTypes = 0,
        .externalSemaphoreFeatures = 0
    };

    // semaphore import/export support
    // we import kms_fence_fd as semaphore and add that as wait semaphore
    // to a render submission so that we only render a buffer when
    // kms signals that it's finished with it.
    // we alos export the semaphore for our render submission as sync_fd
    // and pass that as render_fence_fd to the kernel, signaling
    // that the buffer can only be used when that semaphore is signaled,
    // i.e. we are finished with rendering and all barriers.
    vkGetPhysicalDeviceExternalSemaphoreProperties(
        phdev,
        &(VkPhysicalDeviceExternalSemaphoreInfo) {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
            .pNext = NULL,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
        },
        &props
    );

    if ((props.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) == 0) {
        error("Vulkan can't import drm syncobj fd semaphores");
        abort();
    }

    if ((props.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) == 0) {
        error("Vulkan can't export drm syncobj fd semaphores");
        abort();
    }

    vk_dev->api.getSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR) vkGetDeviceProcAddr(vk_dev->dev, "vkGetSemaphoreFdKHR");
    if (!vk_dev->api.getSemaphoreFdKHR) {
        error("Failed to retrieve vkGetSemaphoreFdKHR\n");
        abort();
    }

    vk_dev->api.importSemaphoreFdKHR = (PFN_vkImportSemaphoreFdKHR) vkGetDeviceProcAddr(vk_dev->dev, "vkImportSemaphoreFdKHR");
    if (!vk_dev->api.importSemaphoreFdKHR) {
        error("Failed to retrieve vkImportSemaphoreFdKHR\n");
        abort();
    }

    vk_dev->api.getMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR) vkGetDeviceProcAddr(vk_dev->dev, "vkGetMemoryFdPropertiesKHR");
    if (!vk_dev->api.getMemoryFdPropertiesKHR) {
        error("Failed to retrieve required vkGetMemoryFdPropertiesKHR\n");
        abort();
    }

    // init renderpass and pipeline
    if (!init_pipeline(vk_dev)) {
        goto error;
    }

    device->vk_device = vk_dev;
    return vk_dev;

error:
    vk_device_destroy(vk_dev);
    return NULL;
}

bool output_vulkan_setup(struct output *output) {
    struct vk_device *vk_dev = output->device->vk_device;
    assert(vk_dev);
    VkResult res;

    output->explicit_fencing = true;

    if (output->num_modifiers == 0) {
        error("Output doesn't support any modifiers, vulkan requires modifiers\n");
        return false;
    }

    // check format support
    // we simply iterate over all the modifiers supported by drm (stored
    // in output) and query with vulkan if the modifier can be used
    // for rendering via vkGetPhysicalDeviceImageFormatProperties2.
    // We are allowed to query it this way (even for modifiers the driver
    // doesn't even know), the function will simply return format_not_supported
    // when it doesn't support/know the modifier.
    // - input -
    VkPhysicalDeviceImageDrmFormatModifierInfoEXT modi = {0};
    modi.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT;

    VkPhysicalDeviceExternalImageFormatInfo efmti = {0};
    efmti.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
    efmti.pNext = &modi;
    efmti.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    VkPhysicalDeviceImageFormatInfo2 fmti = {0};
    fmti.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    fmti.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    fmti.type = VK_IMAGE_TYPE_2D;
    fmti.format = format;
    fmti.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    fmti.pNext = &efmti;

    // - output -
    VkExternalImageFormatProperties efmtp = {0};
    efmtp.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;

    VkImageFormatProperties2 ifmtp = {0};
    ifmtp.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    ifmtp.pNext = &efmtp;

    // supported modifiers
    uint32_t smod_count = 0;
    uint64_t *smods = calloc(output->num_modifiers, sizeof(*smods));
    assert(smods);
    for (unsigned i = 0u; i < output->num_modifiers; ++i) {
        uint64_t mod = output->modifiers[i];
        if (mod != DRM_FORMAT_MOD_LINEAR && mod != DRM_FORMAT_MOD_BROADCOM_UIF) {
            continue;
        }

        modi.drmFormatModifier = mod;
        res = vkGetPhysicalDeviceImageFormatProperties2(vk_dev->phdev, &fmti, &ifmtp);
        if (res == VK_ERROR_FORMAT_NOT_SUPPORTED) {
            continue;
        } else if (res != VK_SUCCESS) {
            vk_error(res, "vkGetPhysicalDeviceImageFormatProperties2");
            return false;
        }

        // we need dmabufs with the given format and modifier to be importable
        // otherwise we can't use the modifier
        if ((efmtp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0) {
            debug("KMS modifier %"PRIu64" not supported by vulkan (2)\n", mod);
            continue;
        }

        smods[smod_count++] = mod;
        debug("Vulkan and KMS support modifier %"PRIu64"\n", mod);

        // we could check/store ifmtp.maxExtent but it should
        // be enough. Otherwise the gpu is connected to an output
        // it can't power on full resolution
    }

    if (smod_count == 0) {
        error("No modifier supported by kms and vulkan");
        return false;
    }

    free(output->modifiers);
    output->num_modifiers = smod_count;
    output->modifiers = smods;

    return true;
}

struct gbm_bo *gbm_bo_create_custom(struct gbm_device *device, int width, int height, int offset, int pitch, size_t size, uint32_t drm_format, uint64_t modifier) {
    struct gbm_bo *bo;
    int fd;
    
    bo = gbm_bo_create_with_modifiers(device, size, 1, GBM_FORMAT_R8, &modifier, 1);
    if (bo == NULL) {
        fprintf(stderr, "Could not allocate graphics memory. gbm_bo_create_with_modifiers: %s\n", strerror(errno));
        return NULL;
    }

    fd = gbm_bo_get_fd(bo);
    if (fd < 0) {
        fprintf(stderr, "Could not get dmabuf fd for graphics memory. gbm_bo_get_fd: %s\n", strerror(errno));
        goto fail_destroy_bo;
    }

    gbm_bo_destroy(bo);
    bo = NULL;

    bo = gbm_bo_import(
        device,
        GBM_BO_IMPORT_FD_MODIFIER,
        &(struct gbm_import_fd_modifier_data) {
            .width = width,
            .height = height,
            .format = drm_format,
            .num_fds = 1,
            .fds = {fd, 0, 0, 0},
            .strides = {pitch, 0, 0, 0},
            .offsets = {offset, 0, 0, 0},
            .modifier = modifier
        },
        GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING // not really made use of 
    );
    if (bo == NULL) {
        fprintf(stderr, "Could not import allocated graphics memory as a gbm bo. gbm_bo_import: %s\n", strerror(errno));
        goto fail_close_fd;
    }

    /// TODO: Should we close the fd here?

    return bo;


    fail_close_fd:
    close(fd);
    // we've already destroyed the bo at this point
    return NULL;

    fail_destroy_bo:
    gbm_bo_destroy(bo);
    return NULL;
}

struct buffer *buffer_vk_create(struct device *device, struct output *output) {
    struct vk_device *vk_dev;
    struct vk_image *img;
    VkDescriptorSet decriptor_set;
    VkCommandBuffer command_buffer;
    VkDeviceMemory img_device_memory, ubo_memory;
    struct gbm_bo *bo;
    VkFramebuffer framebuffer;
    VkSemaphore buffer_semaphore, render_semaphore;
    VkFence render_fence;
    VkImageView img_view;
    VkBuffer ubo;
    uint64_t modifier;
    uint32_t drm_format, gbm_format;
    VkResult ok;
    VkImage vk_img;
    void *ubo_mapped;
    int fd, width, height;

    vk_dev = device->vk_device;
    assert(vk_dev);

    img = malloc(sizeof *img);
    assert(img);

    width = output->mode.hdisplay;
    height = output->mode.vdisplay;
    drm_format = DRM_FORMAT_XRGB8888;
    gbm_format = GBM_FORMAT_XRGB8888;
    modifier = DRM_FORMAT_MOD_LINEAR;

    bool disjoint = false;
    ok = vkCreateImage(
        vk_dev->dev,
        &(VkImageCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .flags = disjoint ? VK_IMAGE_CREATE_DISJOINT_BIT : 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {
                .width = width,
                .height = height,
                .depth = 1
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = 0,
            .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
            .pNext = &(VkExternalMemoryImageCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                .pNext = &(VkImageDrmFormatModifierExplicitCreateInfoEXT) {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
                    .drmFormatModifierPlaneCount = 1,
                    .drmFormatModifier = modifier,
                    .pPlaneLayouts = (VkSubresourceLayout[1]) {
                        {
                            .offset = 0,
                            .size = 0,
                            .rowPitch = 0,
                            .arrayPitch = 0,
                            .depthPitch = 0,
                        }
                    }
                }
            },
        },
        NULL,
        &vk_img
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "Could not create Vulkan image. vkCreateImage");
        abort();
    }

    VkSubresourceLayout layout;
    vkGetImageSubresourceLayout(
        vk_dev->dev,
        vk_img,
        &(VkImageSubresource) {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, // For v3dv, this doesn't really matter
            .mipLevel = 0,
            .arrayLayer = 0
        },
        &layout
    );

    bo = gbm_bo_create_custom(device->gbm_device, width, height, layout.offset, layout.rowPitch, layout.size, gbm_format, modifier);
    assert(bo);

    fd = gbm_bo_get_fd(bo);
    assert(fd >= 0);
    
    // find out as which memory types we can import our dmabuf fd
    VkMemoryFdPropertiesKHR fd_memory_props = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
        .pNext = NULL,
        .memoryTypeBits = 0
    };
    ok = vk_dev->api.getMemoryFdPropertiesKHR(vk_dev->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fd_memory_props);
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkGetMemoryFdPropertiesKHR");
        abort();
    }

    // Find out the memory requirements for our image (the supported memory types for import)
    VkMemoryRequirements2 memr = {0};
    memr.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    vkGetImageMemoryRequirements2(
        vk_dev->dev,
        &(VkImageMemoryRequirementsInfo2) {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
            .image = vk_img,
            .pNext = NULL
        },
        &memr
    );

    // Find a memory type that fits both to the dmabuf and the image
    int mem = find_mem_type(vk_dev->phdev, 0, memr.memoryRequirements.memoryTypeBits & fd_memory_props.memoryTypeBits);
    if (mem < 0) {
        error("no valid memory type index");
        abort();
    }

    // now, create a VkDeviceMemory instance from our dmabuf.
    ok = vkAllocateMemory(
        vk_dev->dev,
        &(VkMemoryAllocateInfo) {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = layout.size,
            .memoryTypeIndex = mem,
            .pNext = &(VkImportMemoryFdInfoKHR) {
                .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                .fd = fd,
                .pNext = &(VkMemoryDedicatedAllocateInfo) {
                    .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                    .image = vk_img,
                    .buffer = VK_NULL_HANDLE,
                    .pNext = NULL
                }
            }
        },
        NULL,
        &img_device_memory
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkAllocateMemory failed");
        abort();
    }

    ok = vkBindImageMemory2(
        vk_dev->dev,
        1,
        &(VkBindImageMemoryInfo) {
            .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
            .image = vk_img,
            .memory = img_device_memory,
            .memoryOffset = 0,
            .pNext = NULL,
        }
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkBindMemory failed");
        abort();
    }

    // create image view and framebuffer for imported image
    ok = vkCreateImageView(
        vk_dev->dev,
        &(VkImageViewCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .flags = 0,
            .image = vk_img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY
            },
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseArrayLayer = 0,
                .layerCount = 1,
                .baseMipLevel = 0,
                .levelCount = 1
            },
            .pNext = NULL,
        },
        NULL,
        &img_view
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkCreateImageView failed");
        abort();
    }

    ok = vkCreateFramebuffer(
        vk_dev->dev,
        &(VkFramebufferCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .flags = 0,
            .renderPass = vk_dev->rp,
            .attachmentCount = 1,
            .pAttachments = &img_view,
            .width = width,
            .height = height,
            .layers = 1,
            .pNext = NULL
        },
        NULL,
        &framebuffer
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkCreateFramebuffer");
        abort();
    }

    const float ubo_size = 4;
    ok = vkCreateBuffer(
        vk_dev->dev,
        &(VkBufferCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .flags = 0,
            .size = ubo_size,
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = NULL,
            .pNext = NULL
        },
        NULL,
        &ubo
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkCreateBuffer");
        abort();
    }

    VkMemoryRequirements ubo_memory_requirements = {0};
    vkGetBufferMemoryRequirements(vk_dev->dev, ubo, &ubo_memory_requirements);

    // the vulkan spec guarantees that non-sparse buffers can
    // always be allocated on host visible, coherent memory, i.e.
    // we must find a valid memory type.
    int mem_type = find_mem_type(
        vk_dev->phdev,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        ubo_memory_requirements.memoryTypeBits
    );
    assert(mem_type >= 0);
    
    ok = vkAllocateMemory(
        vk_dev->dev,
        &(VkMemoryAllocateInfo) {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = ubo_memory_requirements.size,
            .memoryTypeIndex = mem_type,
            .pNext = NULL
        },
        NULL,
        &ubo_memory
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkAllocateMemory");
        abort();
    }

    ok = vkBindBufferMemory(vk_dev->dev, ubo, ubo_memory, 0);
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkBindBufferMemory");
        abort();
    }

    ok = vkMapMemory(vk_dev->dev, ubo_memory, 0, ubo_size, 0, &ubo_mapped);
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkMapMemory");
        abort();
    }

    ok = vkAllocateDescriptorSets(
        vk_dev->dev,
        &(VkDescriptorSetAllocateInfo) {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = vk_dev->ds_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &vk_dev->ds_layout,
            .pNext = NULL
        },
        &decriptor_set
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkAllocateDescriptorSets");
        abort();
    }

    vkUpdateDescriptorSets(
        vk_dev->dev,
        1,
        &(VkWriteDescriptorSet) {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = decriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pImageInfo = 0,
            .pBufferInfo = &(VkDescriptorBufferInfo) {
                .buffer = ubo,
                .offset = 0,
                .range = ubo_size,
            },
            .pTexelBufferView = 0,
            .pNext = NULL,
        },
        0,
        NULL
    );

    // create and record render command buffer
    ok = vkAllocateCommandBuffers(
        vk_dev->dev,
        &(VkCommandBufferAllocateInfo) {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = vk_dev->command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1u,
            .pNext = NULL,
        },
        &command_buffer
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkAllocateCommandBuffers");
        abort();
    }

    vkBeginCommandBuffer(
        command_buffer,
        &(VkCommandBufferBeginInfo) {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
            .pInheritanceInfo = 0,
            .pNext = NULL,
        }
    );

    // we don't need a pipeline barrier for our host write
    // to the mapped ubo here (that happens every frame) because
    // vkQueueSubmit implicitly inserts such a dependency

    // acquire ownership of the image we want to render
    // XXX: as already mentioned on device creation, strictly
    // speaking we need queue_family_foreign here. But since that
    // isn't supported on any mesa driver yet (not even a pr) we
    // try our luck with queue_family_external (which should work for
    // same gpu i guess?). But again: THIS IS NOT GUARANTEED TO WORK,
    // THE STANDARD DOESN'T SUPPORT IT. JUST A TEMPORARY DROP-IN UNTIL
    // THE REAL THING IS SUPPORTED
    uint32_t queue_family = VK_QUEUE_FAMILY_EXTERNAL;

    // TODO: not completely sure about the stages for image ownership transfer
    vkCmdPipelineBarrier(
        command_buffer,
        /* srcStageMask */ VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        /* dstStageMask */ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        /* dependencyFlags */ 0,
        /* memoryBarrierCount */ 0,
        /* pMemoryBarriers */ NULL,
        /* bufferMemoryBarrierCount */ 0,
        /* pBufferMemoryBarriers */ NULL,
        /* imageMemoryBarrierCount */ 1,
        &(VkImageMemoryBarrier) {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL, // doesn't matter really
            .srcQueueFamilyIndex = queue_family,
            .dstQueueFamilyIndex = vk_dev->queue_family,
            .image = vk_img,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .pNext = NULL
        }
    );

    // Renderpass currently specifies don't care as loadOp (since we
    // render the full framebuffer anyways), so we don't need
    // clear values
    // VkClearValue clear_value;
    // clear_value.color.float32[0] = 0.1f;
    // clear_value.color.float32[1] = 0.1f;
    // clear_value.color.float32[2] = 0.1f;
    // clear_value.color.float32[3] = 1.f;

    VkRect2D rect = {
        {0, 0},
        {width, height}
    };
    vkCmdBeginRenderPass(
        command_buffer,
        &(VkRenderPassBeginInfo) {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = vk_dev->rp,
            .framebuffer = framebuffer,
            .renderArea = rect,
            .clearValueCount = 0 /* 1 */,
            .pClearValues = NULL /* &clear_value */,
            .pNext = NULL
        },
        VK_SUBPASS_CONTENTS_INLINE
    );

    vkCmdSetViewport(
        command_buffer,
        /* firstViewport */ 0,
        /* viewportCount */ 1,
        &(VkViewport) {
            .x = 0.f,
            .y = 0.f,
            .width = (float) width,
            .height = (float) height,
            .minDepth = 0.f,
            .maxDepth = 1.f
        }
    );

    vkCmdSetScissor(command_buffer, 0, 1, &rect);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_dev->pipe);
    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        vk_dev->pipe_layout,
        0,
        1, &decriptor_set,
        0, NULL
    );
    vkCmdDraw(command_buffer, 4, 1, 0, 0);

    vkCmdEndRenderPass(command_buffer);

    // release ownership of the image we want to render
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0,
        NULL,
        0,
        NULL,
        1,
        &(VkImageMemoryBarrier) {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL, // doesn't matter really
            .srcQueueFamilyIndex = vk_dev->queue_family,
            .dstQueueFamilyIndex = queue_family,
            .image = vk_img,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .pNext = NULL
        }
    );

    vkEndCommandBuffer(command_buffer);

    // create semaphore that will be used for importing bufer->kms_fence_fd
    // (will be signaled by KMS when the buffer is scanned out on screen,
    // NOT when the buffer is not shown on screen anymore)
    ok = vkCreateSemaphore(
        vk_dev->dev,
        &(VkSemaphoreCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = 0,
            .pNext = NULL
        },
        NULL,
        &buffer_semaphore
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkCreateSemaphore");
        abort();
    }

    ok = vkCreateFence(
        vk_dev->dev,
        &(VkFenceCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0
        },
        NULL,
        &render_fence
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkCreateFence");
        abort();
    }

    // create render semaphore (will be signaled by GPU when rendering is done)
    ok = vkCreateSemaphore(
        vk_dev->dev,
        &(VkSemaphoreCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = 0,
            .pNext = &(VkExportSemaphoreCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
                .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
                .pNext = NULL
            },
        },
        NULL,
        &render_semaphore
    );
    if (ok != VK_SUCCESS) {
        vk_error(ok, "vkCreateSemaphore");
        abort();
    }

    img->memories[0] = img_device_memory;
    img->image = vk_img;
    img->image_view = img_view;
    img->cb = command_buffer;
    img->fb = framebuffer;
    img->ubo = ubo;
    img->ubo_mem = ubo_memory;
    img->ubo_map = ubo_mapped;
    img->ds = decriptor_set;
    img->buffer_semaphore = buffer_semaphore;
    img->render_semaphore = render_semaphore;
    img->render_fence = render_fence;
    img->buffer.output = output;
    img->buffer.in_use = false;
    img->buffer.gem_handles[0] = gbm_bo_get_handle_for_plane(bo, 0).u32;
    img->buffer.gem_handles[1] = 0;
    img->buffer.gem_handles[2] = 0;
    img->buffer.gem_handles[3] = 0;
    img->buffer.fb_id = 0;
    img->buffer.render_fence_fd = -1;
    img->buffer.kms_fence_fd = -1;
    img->buffer.format = drm_format;
    img->buffer.modifier = modifier;
    img->buffer.dumb.mem = NULL;
    img->buffer.dumb.size = 0;
    img->buffer.gbm.bo = bo;
    img->buffer.gbm.img = 0;
    img->buffer.gbm.tex_id = 0;
    img->buffer.gbm.fbo_id = 0;
    img->buffer.width = gbm_bo_get_width(bo);
    img->buffer.height = gbm_bo_get_height(bo);
    img->buffer.pitches[0] = gbm_bo_get_stride_for_plane(bo, 0);
    img->buffer.pitches[1] = 0;
    img->buffer.pitches[2] = 0;
    img->buffer.pitches[3] = 0;
    img->buffer.offsets[0] = gbm_bo_get_offset(bo, 0);
    img->buffer.offsets[1] = 0;
    img->buffer.offsets[2] = 0;
    img->buffer.offsets[3] = 0;

    return &img->buffer;
}

void buffer_vk_destroy(struct device *device, struct buffer *buffer) {
    struct vk_image *img = (struct vk_image *)buffer;
    struct vk_device *vk_dev = device->vk_device;
    if (!vk_dev) {
        error("Expected vk_device in device");
        return;
    }

    VkResult res;
    if (img->render_fence) {
        if (!img->first) {
            res = vkWaitForFences(vk_dev->dev, 1, &img->render_fence, false, UINT64_MAX);
            if (res != VK_SUCCESS) {
                vk_error(res, "vkWaitForFences");
            }
        }

        vkDestroyFence(vk_dev->dev, img->render_fence, NULL);
    }

    // no need to free command buffer or descriptor sets, we will destroy
    // the pools and implicitly free them

    if (img->buffer_semaphore) {
        vkDestroySemaphore(vk_dev->dev, img->buffer_semaphore, NULL);
    }
    if (img->render_semaphore) {
        vkDestroySemaphore(vk_dev->dev, img->render_semaphore, NULL);
    }
    if (img->fb) {
        vkDestroyFramebuffer(vk_dev->dev, img->fb, NULL);
    }
    if (img->image_view) {
        vkDestroyImageView(vk_dev->dev, img->image_view, NULL);
    }
    if (img->image) {
        vkDestroyImage(vk_dev->dev, img->image, NULL);
    }
    if (img->ubo) {
        vkDestroyBuffer(vk_dev->dev, img->ubo, NULL);
    }
    if (img->ubo_mem) {
        vkFreeMemory(vk_dev->dev, img->ubo_mem, NULL);
    }

    for (unsigned i = 0u; i < 4u; ++i) {
        if (img->memories[i]) {
            // will implicitly be unmapped
            // TODO: this currently gives a segmentation fault in
            // the validation layers, probably an error there
            // so not doing it here is the cause for the validation layers
            // to complain about not destroyed memory at the moment
            // vkFreeMemory(vk_dev->dev, img->memories[i], NULL);
        }
    }
    if (img->buffer.gbm.bo) {
        gbm_bo_destroy(img->buffer.gbm.bo);
    }
}

bool buffer_vk_fill(struct buffer *buffer, int frame_num) {
    VkPipelineStageFlags stage;
    struct vk_device *vk_dev;
    struct vk_image *img;
    VkSubmitInfo submission;
    VkResult res;
    int ok;

    img = (struct vk_image*) buffer;
    vk_dev = buffer->output->device->vk_device;

    assert(vk_dev);
    (void) frame_num;

    // update frame number in mapped memory
    *(float*)img->ubo_map = ((float)frame_num) / NUM_ANIM_FRAMES;

    // make the validation layers happy and assert that the command
    // buffer really has finished. Otherwise it's an error in the drm
    // subsystem/an error in our program (buffer reuse) logic
    if (!img->first) {
        res = vkGetFenceStatus(vk_dev->dev, img->render_fence);
        if (res != VK_SUCCESS) {
            vk_error(res, "Invalid render_fence status");
        }

        res = vkResetFences(vk_dev->dev, 1, &img->render_fence);
        if (res != VK_SUCCESS) {
            vk_error(res, "vkResetFences");
        }
    } else {
        img->first = false;
    }

    // submit the buffers command buffer
    // for explicit fencing:
    // - it waits for the kms_fence_fd semaphore
    // - upon completion, it signals the render semaphore
    stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    
    memset(&submission, 0, sizeof(submission));
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &img->cb;

    // we don't have to recreate it every frame but there
    // are currently validation layer errors for sync_fd handles
    // (don't reset payload on export) so we recreate the
    // semaphore in every frame. Shouldn't hurt performance.
    if (img->buffer_semaphore) {
        vkDestroySemaphore(vk_dev->dev, img->render_semaphore, NULL);
    }

    res = vkCreateSemaphore(
        vk_dev->dev,
        &(VkSemaphoreCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = 0,
            .pNext = &(VkExportSemaphoreCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
                .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
                .pNext = NULL
            },
        },
        NULL,
        &img->render_semaphore
    );
    if (res != VK_SUCCESS) {
        vk_error(res, "vkCreateSemaphore");
        return false;
    }

    uint32_t syncobj_handle;
    int syncobj_fd;

    bool has_in_fence;
    if (buffer->kms_fence_fd != -1) {
        ok = drmSyncobjCreate(buffer->output->device->kms_fd, 0, &syncobj_handle);
        if (ok < 0) {
            fprintf(stderr, "Couldn't create syncobj for importing KMS out_fence into vulkan. drmSyncobjCreate: %s\n", strerror(errno));
            abort();
        }

        ok = drmSyncobjImportSyncFile(buffer->output->device->kms_fd, syncobj_handle, buffer->kms_fence_fd);
        if (ok < 0) {
            fprintf(stderr, "Couldn't create import KMS out_fence into syncobj. drmSyncobjImportSyncFile: %s\n", strerror(errno));
            abort();
        }

        
        ok = drmSyncobjHandleToFD(buffer->output->device->kms_fd, syncobj_handle, &syncobj_fd);
        if (ok < 0) {
            fprintf(stderr, "Couldn't export syncobj as fd. drmSyncobjHandleToFD: %s\n", strerror(errno));
            abort();
        }

        // importing semaphore transfers ownership to vulkan
        // importing it as temporary (which is btw the only supported way
        // for sync_fd semaphores) means that after the next wait operation,
        // the semaphore is reset to its prior state, i.e. we can import
        // a new semaphore next frame.
        // As mentioned in the egl backend, the whole kms_fence_fd
        // is not needed with the current architecture of the application
        // since we only re-use buffers after kms is finished with them.
        // In real applications it might be useful though to use it.
        res = vk_dev->api.importSemaphoreFdKHR(
            vk_dev->dev,
            &(VkImportSemaphoreFdInfoKHR) {
                .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
                .semaphore = img->buffer_semaphore,
                .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
                .fd = syncobj_fd
            }
        );
        if (res != VK_SUCCESS) {
            vk_error(res, "vkImportSemaphoreFdKHR");
            abort();
        }

        has_in_fence = true;
    } else {
        has_in_fence = false;
    }

    res = vkQueueSubmit(
        vk_dev->queue,
        1,
        &(VkSubmitInfo) {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,

            .waitSemaphoreCount = has_in_fence ? 1u : 0u,
            .pWaitSemaphores = has_in_fence ? &img->buffer_semaphore : NULL,
            .pWaitDstStageMask = (VkPipelineStageFlags[1]) {
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            },

            .commandBufferCount = 1u,
            .pCommandBuffers = &img->cb,
            
            .signalSemaphoreCount = 1u,
            .pSignalSemaphores = &img->render_semaphore,
            
            .pNext = NULL,
        },
        img->render_fence
    );
    if (res != VK_SUCCESS) {
        vk_error(res, "vkQueueSubmit");
        return false;
    }

    if (img->buffer.render_fence_fd) {
        close(img->buffer.render_fence_fd);
    }

    img->buffer.render_fence_fd = -1;
    // We have to export the fence/semaphore *every frame* since
    // we pass ownership to the kernel when passing the sync_fd.
    // additionally, to export a fence as sync_fd, it
    // "must be signaled, or have an associated fence signal operation
    // pending execution", since sync_fd has copy transference semantics
    // (see the vulkan spec for more details or importing/exporting
    // fences/semaphores). So it's important that we do this *after* we sumit
    // our command buffer using this fence/semaphore
    res = vk_dev->api.getSemaphoreFdKHR(
        vk_dev->dev,
        &(VkSemaphoreGetFdInfoKHR) {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
            .semaphore = img->render_semaphore,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
            .pNext = NULL
        },
        &syncobj_fd
    );
    if (res != VK_SUCCESS) {
        vk_error(res, "vkGetSemaphoreFdKHR");
        abort();
    }

    ok = drmSyncobjFDToHandle(buffer->output->device->kms_fd, syncobj_fd, &syncobj_handle);
    if (ok < 0) {
        fprintf(stderr, "Couldn't convert syncobj fd to syncobj handle. drmSyncobjFDToHandle: %s\n", strerror(errno));
        abort();
    }

    ok = drmSyncobjWait(buffer->output->device->kms_fd, &syncobj_handle, 1, INT64_MAX, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, NULL);
    if (ok < 0) {
        fprintf(stderr, "Couldn't wait for syncobj submit. drmSyncobjWait: %s\n", strerror(errno));
        abort();
    }

    int syncfile_fd;
    ok = drmSyncobjExportSyncFile(buffer->output->device->kms_fd, syncobj_handle, &syncfile_fd);
    if (ok < 0) {
        fprintf(stderr, "Couldn't export syncfile of syncobj handle. drmSyncobjExportSyncFile: %s\n", strerror(errno));
        abort();
    }

    img->buffer.render_fence_fd = syncfile_fd;

    return true;
}
