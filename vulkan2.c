#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <alloca.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <stddef.h>
#include <sys/time.h>

#include <gbm.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_mode.h>
#include <drm_fourcc.h>
#include <vulkan/vulkan.h>

#include <simple.frag.h>
#include <simple.vert.h>
#include <vkcube.frag.h>
#include <vkcube.vert.h>
#include <modesetting.h>
#include <esUtil.h>

const char *vk_strerror(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
        case VK_ERROR_NOT_PERMITTED_EXT: return "VK_ERROR_NOT_PERMITTED_EXT";
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
        case VK_THREAD_IDLE_KHR: return "VK_THREAD_IDLE_KHR";
        case VK_THREAD_DONE_KHR: return "VK_THREAD_DONE_KHR";
        case VK_OPERATION_DEFERRED_KHR: return "VK_OPERATION_DEFERRED_KHR";
        case VK_OPERATION_NOT_DEFERRED_KHR: return "VK_OPERATION_NOT_DEFERRED_KHR";
        case VK_PIPELINE_COMPILE_REQUIRED_EXT: return "VK_PIPELINE_COMPILE_REQUIRED_EXT";
        default: return "<unknown result code>";
    }
}

#define LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)
#define LOG_VK_ERROR(result, fmt, ...) fprintf(stderr, fmt ": %s\n", __VA_ARGS__ vk_strerror(result));
#define LOG_DEBUG(...) printf(__VA_ARGS__);


struct vkdev {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue graphics_queue;
    VkDebugUtilsMessengerEXT debug_utils_messenger;
    VkCommandPool graphics_cmd_pool;

    PFN_vkCreateDebugUtilsMessengerEXT create_debug_utils_messenger;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_utils_messenger;
};

struct debug_messenger {
    VkDebugUtilsMessengerCreateFlagsEXT flags;
    VkDebugUtilsMessageSeverityFlagsEXT severities;
    VkDebugUtilsMessageTypeFlagsEXT types;
    PFN_vkDebugUtilsMessengerCallbackEXT cb;
    void *userdata;
};

static int get_graphics_queue_family_index(VkPhysicalDevice device) {
    uint32_t n_queue_families;

    vkGetPhysicalDeviceQueueFamilyProperties(device, &n_queue_families, NULL);

    VkQueueFamilyProperties queue_families[n_queue_families];
    vkGetPhysicalDeviceQueueFamilyProperties(device, &n_queue_families, queue_families);

    for (unsigned i = 0; i < n_queue_families; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            return i;
        }
    }

    return -1;
}

static int score_physical_device(VkPhysicalDevice device, const char **required_device_extensions) {
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures features;
    VkResult ok;
    uint32_t n_available_extensions;
    int graphics_queue_fam_index;
    int score = 1;

    vkGetPhysicalDeviceProperties(device, &props);
    vkGetPhysicalDeviceFeatures(device, &features);

    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 15;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        score += 10;
    }

    graphics_queue_fam_index = get_graphics_queue_family_index(device);
    if (graphics_queue_fam_index == -1) {
        LOG_ERROR("Physical device does not support a graphics queue.\n");
        return 0;
    }

    ok = vkEnumerateDeviceExtensionProperties(device, NULL, &n_available_extensions, NULL);
    if (ok != 0) {
        LOG_VK_ERROR(ok, "Could not query available physical device extensions. vkEnumerateDeviceExtensionProperties");
        return 0;
    }

    VkExtensionProperties available_extensions[n_available_extensions];
    ok = vkEnumerateDeviceExtensionProperties(device, NULL, &n_available_extensions, available_extensions);
    if (ok != 0) {
        LOG_VK_ERROR(ok, "Could not query available physical device extensions. vkEnumerateDeviceExtensionProperties");
        return 0;
    }

    for (const char **cursor = required_device_extensions; *cursor != NULL; cursor++) {
        for (unsigned i = 0; i < n_available_extensions; i++) {
            if (strcmp(available_extensions[i].extensionName, *cursor) == 0) {
                goto found;
            }
        }
        LOG_ERROR("Required extension %s is not supported by vulkan device.\n", *cursor);
        return 0;

        found:
        continue;
    }

    return score;
}

struct vkdev *vkdev_new(
    const char *application_name, uint32_t application_version,
    const char *engine_name, uint32_t engine_version,
    uint32_t vulkan_api_version,
    const char **required_layers,
    const char **optional_layers,
    const char **required_instance_extensions,
    const char **optional_instance_extensions,
    const char **required_device_extensions,
    const char **optional_device_extensions,
    const struct debug_messenger *messenger
) {
    PFN_vkCreateDebugUtilsMessengerEXT create_debug_utils_messenger;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_utils_messenger;
    VkDebugUtilsMessengerEXT debug_utils_messenger;
    VkCommandPool graphics_cmd_pool;
    struct vkdev *dev;
    VkInstance instance;
    VkDevice device;
    VkResult ok;
    VkQueue graphics_queue;
    uint32_t n_available_layers, n_available_instance_extensions, n_available_device_extensions, n_physical_devices;
    int n_layers, n_instance_extensions, n_device_extensions;
    int graphics_queue_family_index;

    ok = vkEnumerateInstanceLayerProperties(&n_available_layers, NULL);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not query instance layers. vkEnumerateInstanceLayerProperties");
        return NULL;
    }

    VkLayerProperties *available_layers = alloca(sizeof(VkLayerProperties) * n_available_layers);
    ok = vkEnumerateInstanceLayerProperties(&n_available_layers, available_layers);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not query instance layers. vkEnumerateInstanceLayerProperties");
        return NULL;
    }
    
    n_layers = 0;
    const char **layers = alloca(sizeof(const char*) * n_available_layers);
    for (const char **cursor = required_layers; cursor != NULL && *cursor != NULL; cursor++) {
        for (unsigned i = 0; i < n_available_layers; i++) {
            if (strcmp(available_layers[i].layerName, *cursor) == 0) {
                layers[n_layers] = *cursor;
                n_layers++;
                goto found_required_layer;
            }
        }
        LOG_ERROR("Required layer %s is not supported by vulkan instance.\n", *cursor);
        return NULL;

        found_required_layer:
        continue;
    }

    for (const char **cursor = optional_layers; cursor != NULL && *cursor != NULL; cursor++) {
        for (unsigned i = 0; i < n_available_layers; i++) {
            if (strcmp(available_layers[i].layerName, *cursor) == 0) {
                layers[n_layers] = *cursor;
                n_layers++;
                goto found_optional_layer;
            }
        }
        LOG_ERROR("Optional layer %s is not supported by vulkan instance.\n", *cursor);

        found_optional_layer:
        continue;
    }

    ok = vkEnumerateInstanceExtensionProperties(NULL, &n_available_instance_extensions, NULL);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not query instance extensions. vkEnumerateInstanceExtensionProperties");
        return NULL;
    }

    VkExtensionProperties *available_instance_extensions = alloca(sizeof(VkExtensionProperties) * n_available_instance_extensions);
    ok = vkEnumerateInstanceExtensionProperties(NULL, &n_available_instance_extensions, available_instance_extensions);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not query instance extensions. vkEnumerateInstanceExtensionProperties");
        return NULL;
    }

    n_instance_extensions = 0;
    const char **instance_extensions = alloca(sizeof(const char*) * n_available_instance_extensions);
    for (const char **cursor = required_instance_extensions; cursor != NULL && *cursor != NULL; cursor++) {
        for (unsigned i = 0; i < n_available_instance_extensions; i++) {
            if (strcmp(available_instance_extensions[i].extensionName, *cursor) == 0) {
                instance_extensions[n_instance_extensions] = *cursor;
                n_instance_extensions++;
                goto found_required_instance_extension;
            }
        }
        LOG_ERROR("Required instance extension %s is not supported by vulkan instance.\n", *cursor);
        return NULL;

        found_required_instance_extension:
        continue;
    }

    for (const char **cursor = optional_instance_extensions; cursor != NULL && *cursor != NULL; cursor++) {
        for (unsigned i = 0; i < n_available_instance_extensions; i++) {
            if (strcmp(available_instance_extensions[i].extensionName, *cursor) == 0) {
                instance_extensions[n_instance_extensions] = *cursor;
                n_instance_extensions++;
                goto found_optional_instance_extension;
            }
        }
        LOG_ERROR("Optional instance extension %s is not supported by vulkan instance.\n", *cursor);

        found_optional_instance_extension:
        continue;
    }

    ok = vkCreateInstance(
        &(VkInstanceCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .flags = 0,
            .pApplicationInfo = &(VkApplicationInfo) {
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pApplicationName = application_name,
                .applicationVersion = application_version,
                .pEngineName = engine_name,
                .engineVersion = engine_version,
                .apiVersion = vulkan_api_version,
                .pNext = NULL,
            },
            .enabledLayerCount = n_layers,
            .ppEnabledLayerNames = layers,
            .enabledExtensionCount = n_instance_extensions,
            .ppEnabledExtensionNames = instance_extensions,
            .pNext = messenger != NULL ? &(VkDebugUtilsMessengerCreateInfoEXT) {
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                .flags = 0,
                .messageSeverity = messenger->severities,
                .messageType = messenger->types,
                .pfnUserCallback = messenger->cb,
                .pUserData = messenger->userdata,
                .pNext = NULL,
            } : NULL,
        },
        NULL,
        &instance
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not create instance. vkCreateInstance");
        return NULL;
    }

    if (messenger != NULL) {
        create_debug_utils_messenger = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (create_debug_utils_messenger == NULL) {
            LOG_ERROR("Could not resolve vkCreateDebugUtilsMessengerEXT function.\n");
            goto fail_destroy_instance;
        }

        destroy_debug_utils_messenger = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy_debug_utils_messenger == NULL) {
            LOG_ERROR("Could not resolve vkDestroyDebugUtilsMessengerEXT function.\n");
            goto fail_destroy_instance;
        }

        ok = create_debug_utils_messenger(
            instance,
            &(VkDebugUtilsMessengerCreateInfoEXT) {
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                .flags = 0,
                .messageSeverity = messenger->severities,
                .messageType = messenger->types,
                .pfnUserCallback = messenger->cb,
                .pUserData = messenger->userdata,
                .pNext = NULL,
            },
            NULL,
            &debug_utils_messenger
        );
        if (ok != VK_SUCCESS) {
            LOG_VK_ERROR(ok, "Could not create debug utils messenger. vkCreateDebugUtilsMessengerEXT");
            goto fail_destroy_instance;
        }
    } else {
        debug_utils_messenger = VK_NULL_HANDLE;
    }

    ok = vkEnumeratePhysicalDevices(instance, &n_physical_devices, NULL);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not enumerate physical devices. vkEnumeratePhysicalDevices");
        goto fail_maybe_destroy_messenger;
    }

    VkPhysicalDevice *physical_devices = alloca(sizeof(VkPhysicalDevice) * n_physical_devices);
    ok = vkEnumeratePhysicalDevices(instance, &n_physical_devices, physical_devices);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not enumerate physical devices. vkEnumeratePhysicalDevices");
        goto fail_maybe_destroy_messenger;
    }

    VkPhysicalDevice best_device = VK_NULL_HANDLE;
    int score = 0;
    for (unsigned i = 0; i < n_physical_devices; i++) {
        VkPhysicalDevice this = physical_devices[i];
        int this_score = score_physical_device(this, required_device_extensions);
        
        if (this_score > score) {
            best_device = this;
            score = this_score;
        }
    }

    if (best_device == VK_NULL_HANDLE) {
        LOG_ERROR("No suitable physical device found.\n");
        goto fail_maybe_destroy_messenger;
    }

    ok = vkEnumerateDeviceExtensionProperties(best_device, NULL, &n_available_device_extensions, NULL);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not query device extensions. vkEnumerateDeviceExtensionProperties");
        return NULL;
    }

    VkExtensionProperties *available_device_extensions = alloca(sizeof(VkExtensionProperties) * n_available_device_extensions);
    ok = vkEnumerateDeviceExtensionProperties(best_device, NULL, &n_available_device_extensions, available_device_extensions);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not query device extensions. vkEnumerateDeviceExtensionProperties");
        return NULL;
    }

    n_device_extensions = 0;
    const char **device_extensions = alloca(sizeof(const char*) * n_available_device_extensions);
    for (const char **cursor = required_device_extensions; cursor != NULL && *cursor != NULL; cursor++) {
        device_extensions[n_device_extensions] = *cursor;
        n_device_extensions++;
    }

    for (const char **cursor = optional_device_extensions; cursor != NULL && *cursor != NULL; cursor++) {
        for (unsigned i = 0; i < n_available_device_extensions; i++) {
            if (strcmp(available_device_extensions[i].extensionName, *cursor) == 0) {
                device_extensions[n_device_extensions] = *cursor;
                n_device_extensions++;
                goto found_optional_device_extension;
            }
        }
        LOG_ERROR("Optional device extension %s is not supported by vulkan device.\n", *cursor);

        found_optional_device_extension:
        continue;
    }

    graphics_queue_family_index = get_graphics_queue_family_index(best_device);

    ok = vkCreateDevice(
        best_device,
        &(const VkDeviceCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = (const VkDeviceQueueCreateInfo[1]) {
                {
                    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    .flags = 0,
                    .queueFamilyIndex = graphics_queue_family_index,
                    .queueCount = 1,
                    .pQueuePriorities = (float[1]) { 1.0f },
                    .pNext = NULL,
                },
            },
            .enabledLayerCount = n_layers,
            .ppEnabledLayerNames = layers,
            .enabledExtensionCount = n_device_extensions,
            .ppEnabledExtensionNames = device_extensions,
            .pEnabledFeatures = &(const VkPhysicalDeviceFeatures) { 0 },
            .pNext = NULL,
        },
        NULL,
        &device
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not create logical device. vkCreateDevice");
        goto fail_maybe_destroy_messenger;
    }

    vkGetDeviceQueue(device, graphics_queue_family_index, 0, &graphics_queue);

    ok = vkCreateCommandPool(
        device,
        &(const VkCommandPoolCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = graphics_queue_family_index,
            .pNext = NULL,
        },
        NULL,
        &graphics_cmd_pool
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not create command pool for allocating graphics command buffers. vkCreateCommandPool");
        goto fail_destroy_device;
    }

    dev = malloc(sizeof *dev);
    if (dev == NULL) {
        goto fail_destroy_graphics_cmd_pool;
    }

    dev->device = device;
    dev->physical_device = best_device;
    dev->instance = instance;
    dev->graphics_queue = graphics_queue;
    dev->debug_utils_messenger = debug_utils_messenger;
    dev->graphics_cmd_pool = graphics_cmd_pool;
    dev->create_debug_utils_messenger = create_debug_utils_messenger;
    dev->destroy_debug_utils_messenger = destroy_debug_utils_messenger;
    return dev;


    fail_destroy_graphics_cmd_pool:
    vkDestroyCommandPool(device, graphics_cmd_pool, NULL);

    fail_destroy_device:
    vkDestroyDevice(device, NULL);

    fail_maybe_destroy_messenger:
    if (debug_utils_messenger != VK_NULL_HANDLE) {
        destroy_debug_utils_messenger(instance, debug_utils_messenger, NULL);
    }

    fail_destroy_instance:
    vkDestroyInstance(instance, NULL);
    return NULL;
}

void vkdev_destroy(struct vkdev *dev) {
    vkDestroyCommandPool(dev->device, dev->graphics_cmd_pool, NULL);
    vkDestroyDevice(dev->device, NULL);
    if (dev->debug_utils_messenger != VK_NULL_HANDLE) {
        dev->destroy_debug_utils_messenger(dev->instance, dev->debug_utils_messenger, NULL);
    }
    vkDestroyInstance(dev->instance, NULL);
    free(dev);
}



struct vk_kms_image {
    struct gbm_bo *bo;
    int width, height;
    uint32_t drm_format, gbm_format;
    uint64_t drm_modifier;
    VkFormat vk_format;
    VkImage image;
    VkDeviceMemory memory;
};

static int find_mem_type(VkPhysicalDevice phdev, VkMemoryPropertyFlags flags, uint32_t req_bits) {
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

static struct vk_kms_image *vk_kms_image_new(
    struct vkdev *dev,
    struct gbm_device *gbm_device,
    int width, int height,
    VkFormat vk_format,
    uint32_t gbm_format,
    uint32_t drm_format, uint64_t drm_modifier
) {
    PFN_vkGetMemoryFdPropertiesKHR get_memory_fd_props;
    VkSubresourceLayout layout;
    struct vk_kms_image *img;
    VkDeviceMemory img_device_memory;
    struct gbm_bo *bo;
    VkResult ok;
    VkImage vkimg;
    int fd;

    ok = vkCreateImage(
        dev->device,
        &(VkImageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = vk_format,
            .extent = { .width = width, .height = height, .depth = 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = 0,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .pNext =
                &(VkExternalMemoryImageCreateInfo){
                    .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                    .pNext =
                        &(VkImageDrmFormatModifierExplicitCreateInfoEXT){
                            .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
                            .drmFormatModifierPlaneCount = 1,
                            .drmFormatModifier = drm_modifier,
                            .pPlaneLayouts =
                                (VkSubresourceLayout[1]){
                                    {
                                        .offset = 0,
                                        .size = 0,
                                        .rowPitch = 0,
                                        .arrayPitch = 0,
                                        .depthPitch = 0,
                                    },
                                },
                        },
                },
        },
        NULL,
        &vkimg
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not create Vulkan image. vkCreateImage");
        return NULL;
    }

    vkGetImageSubresourceLayout(
        dev->device,
        vkimg,
        &(VkImageSubresource){
            .aspectMask = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,  // For v3dv, this doesn't really matter
            .mipLevel = 0,
            .arrayLayer = 0,
        },
        &layout
    );

    bo = gbm_bo_create_with_modifiers(
        gbm_device,
        width,
        height,
        gbm_format,
        &drm_modifier,
        1
    );
    if (bo == NULL) {
        LOG_ERROR("Could not create GBM BO. gbm_bo_create_with_modifiers: %s\n", strerror(errno));
        goto fail_destroy_image;
    }

    if (gbm_bo_get_offset(bo, 0) != layout.offset) {
        LOG_ERROR("GBM BO layout doesn't match image layout. This is probably a driver / kernel bug.\n");
        goto fail_destroy_bo;
    }

    if (gbm_bo_get_stride_for_plane(bo, 0) != layout.rowPitch) {
        LOG_ERROR("GBM BO layout doesn't match image layout. This is probably a driver / kernel bug.\n");
        goto fail_destroy_bo;
    }

    fd = gbm_bo_get_fd(bo);
    if (fd < 0) {
        LOG_ERROR("Couldn't get dmabuf fd for GBM buffer. gbm_bo_get_fd: %s\n", strerror(errno));
        goto fail_destroy_bo;
    }

    // find out as which memory types we can import our dmabuf fd
    VkMemoryFdPropertiesKHR fd_memory_props = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
        .pNext = NULL,
        .memoryTypeBits = 0,
    };

    get_memory_fd_props = (PFN_vkGetMemoryFdPropertiesKHR) vkGetDeviceProcAddr(dev->device, "vkGetMemoryFdPropertiesKHR");
    if (get_memory_fd_props == NULL) {
        LOG_ERROR("Couldn't resolve vkGetMemoryFdPropertiesKHR.\n");
        goto fail_destroy_bo;
    }

    ok = get_memory_fd_props(
        dev->device,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        fd,
        &fd_memory_props
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't get dmabuf memory properties. vkGetMemoryFdPropertiesKHR");
        goto fail_destroy_bo;
    }

    // Find out the memory requirements for our image (the supported memory types for import)
    VkMemoryRequirements2 image_memory_reqs = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .memoryRequirements = { 0 },
        .pNext = NULL
    };

    vkGetImageMemoryRequirements2(
        dev->device,
        &(VkImageMemoryRequirementsInfo2) {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
            .image = vkimg,
            .pNext = NULL,
        },
        &image_memory_reqs
    );

    // Find a memory type that fits both to the dmabuf and the image
    int mem = find_mem_type(dev->physical_device, 0, image_memory_reqs.memoryRequirements.memoryTypeBits & fd_memory_props.memoryTypeBits);
    if (mem < 0) {
        LOG_ERROR("Couldn't find a memory type that's both supported by the image and the dmabuffer.\n");
        goto fail_destroy_bo;
    }

    // now, create a VkDeviceMemory instance from our dmabuf.
    ok = vkAllocateMemory(
        dev->device,
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
                    .image = vkimg,
                    .buffer = VK_NULL_HANDLE,
                    .pNext = NULL,
                },
            },
        },
        NULL,
        &img_device_memory
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't import dmabuf as vulkan device memory. vkAllocateMemory");
        goto fail_destroy_bo;
    }

    ok = vkBindImageMemory2(
        dev->device,
        1,
        &(VkBindImageMemoryInfo){
            .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
            .image = vkimg,
            .memory = img_device_memory,
            .memoryOffset = 0,
            .pNext = NULL,
        }
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't bind dmabuf-backed vulkan device memory to vulkan image. vkBindImageMemory2");
        goto fail_free_device_memory;
    }

    img = malloc(sizeof *img);
    if (img == NULL) {
        goto fail_free_device_memory;
    }

    img->bo = bo;
    img->memory = img_device_memory;
    img->image = vkimg;
    img->width = width;
    img->height = height;
    img->drm_format = drm_format;
    img->gbm_format = gbm_format;
    img->drm_modifier = drm_modifier;
    img->vk_format = vk_format;
    return img;


    fail_free_device_memory:
    vkFreeMemory(dev->device, img_device_memory, NULL);

    fail_destroy_bo:
    gbm_bo_destroy(bo);

    fail_destroy_image:
    vkDestroyImage(dev->device, vkimg, NULL);
    return NULL;
}

static void vk_kms_image_destroy(struct vk_kms_image *img, VkDevice device) {
    vkFreeMemory(device, img->memory, NULL);
    gbm_bo_destroy(img->bo);
    vkDestroyImage(device, img->image, NULL);
    free(img);
}


struct pipeline_fb {
    int width, height;
    VkImageView view;
    VkFramebuffer fb;
};

static struct pipeline_fb *pipeline_fb_new(struct vkdev *dev, struct vk_kms_image *image, VkRenderPass renderpass) {
    struct pipeline_fb *fb;
    VkFramebuffer vkfb;
    VkImageView view;
    VkResult ok;

    ok = vkCreateImageView(
        dev->device,
        &(const VkImageViewCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .flags = 0,
            .image = image->image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = image->vk_format,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .pNext = NULL,
        },
        NULL,
        &view
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not create image view. vkCreateImageView");
        return NULL;
    }

    ok = vkCreateFramebuffer(
        dev->device,
        &(const VkFramebufferCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .flags = 0,
            .renderPass = renderpass,
            .attachmentCount = 1,
            .pAttachments = (VkImageView[]) { view },
            .width = image->width,
            .height = image->height,
            .layers = 1,
            .pNext = NULL,
        },
        NULL,
        &vkfb
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not create framebuffer. vkCreateFramebuffer");
        return NULL;
    }

    fb = malloc(sizeof *fb);
    if (fb == NULL) {
        goto fail_destroy_vkfb;
    }

    fb->fb = vkfb;
    fb->view = view;
    fb->width = image->width;
    fb->height = image->height;
    return fb;


    fail_destroy_vkfb:
    vkDestroyFramebuffer(dev->device, vkfb, NULL);

    fail_destroy_image_view:
    vkDestroyImageView(dev->device, view, NULL);
    return NULL;
}

static void pipeline_fb_destroy(struct pipeline_fb *fb, VkDevice device) {
    vkDestroyFramebuffer(device, fb->fb, NULL);
    vkDestroyImageView(device, fb->view, NULL);
    free(fb);
}


struct cube_ubo_data {
    ESMatrix modelview;
    ESMatrix modelviewprojection;
    float normal[12];
};

struct cube_gpu_data {
    struct cube_ubo_data ubo;
    float vertices[3*4*6];
    float colors[3*4*6];
    float normals[3*4*6];
};

struct cube_gpu_buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDescriptorSet descriptor_set;
    VkDescriptorPool descriptor_pool;
    struct cube_gpu_data *mapped;
};

static struct cube_gpu_buffer *cube_gpu_buffer_new(struct vkdev *dev, VkDescriptorSetLayout ubo_layout) {
    struct cube_gpu_data *mapped;
    VkDescriptorPool descriptor_pool;
    struct cube_gpu_buffer *ubo;
    VkDescriptorSet descriptor_set;
    VkDeviceMemory mem;
    VkBuffer buffer;
    VkResult ok;

    ok = vkCreateBuffer(
        dev->device,
        &(VkBufferCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(struct cube_gpu_data),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .flags = 0
        },
        NULL,
        &buffer
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't create uniform / vertex buffer. vkCreateBuffer");
        return NULL;
    }

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(dev->device, buffer, &reqs);

    VkDeviceSize mem_size = reqs.size < sizeof(struct cube_gpu_data) ? sizeof(struct cube_gpu_data) : reqs.size;

    int memory_type = find_mem_type(dev->physical_device, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, reqs.memoryTypeBits);
    if (memory_type < 0) {
        LOG_ERROR("Couldn't find a memory type that is accessible from host and coherent.\n");
        goto fail_destroy_buffer;
    }

    ok = vkAllocateMemory(
        dev->device,
        &(const VkMemoryAllocateInfo) {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mem_size,
            .memoryTypeIndex = memory_type,
            .pNext = NULL
        },
        NULL,
        &mem
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't allocate memory for uniform / vertex buffer. vkAllocateMemory");
        goto fail_destroy_buffer;
    }

    ok = vkMapMemory(dev->device, mem, 0, mem_size, 0, (void**) &mapped);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't map uniform / vertex buffer. vkMapMemory");
        goto fail_free_device_mem;
    }

    // clang-format off
    static const float vertices[] = {
        // front
        -1.0f, -1.0f, +1.0f, // point blue
        +1.0f, -1.0f, +1.0f, // point magenta
        -1.0f, +1.0f, +1.0f, // point cyan
        +1.0f, +1.0f, +1.0f, // point white
        // back
        +1.0f, -1.0f, -1.0f, // point red
        -1.0f, -1.0f, -1.0f, // point black
        +1.0f, +1.0f, -1.0f, // point yellow
        -1.0f, +1.0f, -1.0f, // point green
        // right
        +1.0f, -1.0f, +1.0f, // point magenta
        +1.0f, -1.0f, -1.0f, // point red
        +1.0f, +1.0f, +1.0f, // point white
        +1.0f, +1.0f, -1.0f, // point yellow
        // left
        -1.0f, -1.0f, -1.0f, // point black
        -1.0f, -1.0f, +1.0f, // point blue
        -1.0f, +1.0f, -1.0f, // point green
        -1.0f, +1.0f, +1.0f, // point cyan
        // top
        -1.0f, +1.0f, +1.0f, // point cyan
        +1.0f, +1.0f, +1.0f, // point white
        -1.0f, +1.0f, -1.0f, // point green
        +1.0f, +1.0f, -1.0f, // point yellow
        // bottom
        -1.0f, -1.0f, -1.0f, // point black
        +1.0f, -1.0f, -1.0f, // point red
        -1.0f, -1.0f, +1.0f, // point blue
        +1.0f, -1.0f, +1.0f  // point magenta
    };

    static const float colors[] = {
        // front
        0.0f,  0.0f,  1.0f, // blue
        1.0f,  0.0f,  1.0f, // magenta
        0.0f,  1.0f,  1.0f, // cyan
        1.0f,  1.0f,  1.0f, // white
        // back
        1.0f,  0.0f,  0.0f, // red
        0.0f,  0.0f,  0.0f, // black
        1.0f,  1.0f,  0.0f, // yellow
        0.0f,  1.0f,  0.0f, // green
        // right
        1.0f,  0.0f,  1.0f, // magenta
        1.0f,  0.0f,  0.0f, // red
        1.0f,  1.0f,  1.0f, // white
        1.0f,  1.0f,  0.0f, // yellow
        // left
        0.0f,  0.0f,  0.0f, // black
        0.0f,  0.0f,  1.0f, // blue
        0.0f,  1.0f,  0.0f, // green
        0.0f,  1.0f,  1.0f, // cyan
        // top
        0.0f,  1.0f,  1.0f, // cyan
        1.0f,  1.0f,  1.0f, // white
        0.0f,  1.0f,  0.0f, // green
        1.0f,  1.0f,  0.0f, // yellow
        // bottom
        0.0f,  0.0f,  0.0f, // black
        1.0f,  0.0f,  0.0f, // red
        0.0f,  0.0f,  1.0f, // blue
        1.0f,  0.0f,  1.0f  // magenta
    };

    static const float normals[] = {
        // front
        +0.0f, +0.0f, +1.0f, // forward
        +0.0f, +0.0f, +1.0f, // forward
        +0.0f, +0.0f, +1.0f, // forward
        +0.0f, +0.0f, +1.0f, // forward
        // back
        +0.0f, +0.0f, -1.0f, // backbard
        +0.0f, +0.0f, -1.0f, // backbard
        +0.0f, +0.0f, -1.0f, // backbard
        +0.0f, +0.0f, -1.0f, // backbard
        // right
        +1.0f, +0.0f, +0.0f, // right
        +1.0f, +0.0f, +0.0f, // right
        +1.0f, +0.0f, +0.0f, // right
        +1.0f, +0.0f, +0.0f, // right
        // left
        -1.0f, +0.0f, +0.0f, // left
        -1.0f, +0.0f, +0.0f, // left
        -1.0f, +0.0f, +0.0f, // left
        -1.0f, +0.0f, +0.0f, // left
        // top
        +0.0f, +1.0f, +0.0f, // up
        +0.0f, +1.0f, +0.0f, // up
        +0.0f, +1.0f, +0.0f, // up
        +0.0f, +1.0f, +0.0f, // up
        // bottom
        +0.0f, -1.0f, +0.0f, // down
        +0.0f, -1.0f, +0.0f, // down
        +0.0f, -1.0f, +0.0f, // down
        +0.0f, -1.0f, +0.0f  // down
    };
    // clang-format on

    memcpy(mapped->vertices, vertices, sizeof(vertices));
    memcpy(mapped->colors, colors, sizeof(colors));
    memcpy(mapped->normals, normals, sizeof(normals));

    ok = vkBindBufferMemory(dev->device, buffer, mem, 0);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't bind device memory to vertex / uniform buffer. vkBindBufferMemory");
        goto fail_unmap_mem;
    }

    ok = vkCreateDescriptorPool(
        dev->device,
        &(const VkDescriptorPoolCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = 0,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &(const VkDescriptorPoolSize) {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1
            },
            .pNext = NULL,
        },
        NULL,
        &descriptor_pool
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't create a descriptor pool for allocating the uniform / vertex buffer descriptor set. vkCreateDescriptorPool");
        goto fail_unmap_mem;
    }

    ok = vkAllocateDescriptorSets(
        dev->device,
        &(const VkDescriptorSetAllocateInfo) {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &ubo_layout,
        },
        &descriptor_set
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't allocate a descriptor set for uniform / vertex buffer. vkAllocateDescriptorSets");
        goto fail_destroy_descriptor_pool;
    }

    vkUpdateDescriptorSets(
        dev->device,
        1,
        (const VkWriteDescriptorSet []) {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pImageInfo = NULL,
                .pBufferInfo = &(VkDescriptorBufferInfo) {
                    .buffer = buffer,
                    .offset = 0,
                    .range = sizeof(struct cube_ubo_data),
                },
                .pTexelBufferView = NULL,
                .pNext = NULL,
            }
        },
        0,
        NULL
    );

    ubo = malloc(sizeof *ubo);
    if (ubo == NULL) {
        goto fail_free_descriptor_set;
    }

    ubo->buffer = buffer;
    ubo->memory = mem;
    ubo->descriptor_pool = descriptor_pool;
    ubo->descriptor_set = descriptor_set;
    ubo->mapped = mapped;
    return ubo;


    fail_free_descriptor_set:
    vkFreeDescriptorSets(dev->device, descriptor_pool, 1, &descriptor_set);

    fail_destroy_descriptor_pool:
    vkDestroyDescriptorPool(dev->device, descriptor_pool, NULL);

    fail_unmap_mem:
    vkUnmapMemory(dev->device, mem);

    fail_free_device_mem:
    vkFreeMemory(dev->device, mem, NULL);

    fail_destroy_buffer:
    vkDestroyBuffer(dev->device, buffer, NULL);
    return NULL;
}

static void cube_gpu_buffer_destroy(struct cube_gpu_buffer *ubo, VkDevice device) {
    vkFreeDescriptorSets(device, ubo->descriptor_pool, 1, &ubo->descriptor_set);
    vkDestroyDescriptorPool(device, ubo->descriptor_pool, NULL);
    vkUnmapMemory(device, ubo->memory);
    vkFreeMemory(device, ubo->memory, NULL);
    vkDestroyBuffer(device, ubo->buffer, NULL);
    free(ubo);
}

static void cube_gpu_buffer_update_transforms(struct cube_gpu_buffer *buf, struct timeval start_time, float aspect_ratio) {
    struct cube_ubo_data ubo;
    struct timeval tv;
    uint64_t t;

    gettimeofday(&tv, NULL);

    t = ((tv.tv_sec * 1000 + tv.tv_usec / 1000) -
            (start_time.tv_sec * 1000 + start_time.tv_usec / 1000)) / 5;

    esMatrixLoadIdentity(&ubo.modelview);
    esTranslate(&ubo.modelview, 0.0f, 0.0f, -8.0f);
    esRotate(&ubo.modelview, 45.0f + (0.25f * t), 1.0f, 0.0f, 0.0f);
    esRotate(&ubo.modelview, 45.0f - (0.5f * t), 0.0f, 1.0f, 0.0f);
    esRotate(&ubo.modelview, 10.0f + (0.15f * t), 0.0f, 0.0f, 1.0f);

    ESMatrix projection;
    esMatrixLoadIdentity(&projection);
    esFrustum(&projection, -2.8f, +2.8f, -2.8f * aspect_ratio, +2.8f * aspect_ratio, 6.0f, 10.0f);

    esMatrixLoadIdentity(&ubo.modelviewprojection);
    esMatrixMultiply(&ubo.modelviewprojection, &ubo.modelview, &projection);

    /* The mat3 normalMatrix is laid out as 3 vec4s. */
    memcpy(&ubo.normal, &ubo.modelview, sizeof(ubo.normal));

    memcpy(&(buf->mapped->ubo), &ubo, sizeof(ubo));
}


struct cube_pipeline {
    VkShaderModule vert_shader, frag_shader;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;
    VkRenderPass renderpass;
    VkPipeline pipeline;
};

static struct cube_pipeline *cube_pipeline_new(struct vkdev *dev, int width, int height, VkFormat format) {
    VkPipelineLayout pipeline_layout;
    struct cube_pipeline *pipeline;
    VkDescriptorSetLayout set_layout;
    VkShaderModule vert_shader, frag_shader;
    VkRenderPass renderpass;
    VkPipeline vkpipeline;
    VkResult ok;

    ok = vkCreateShaderModule(
        dev->device,
        &(const VkShaderModuleCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .flags = 0,
            .codeSize = sizeof(vkcube_vert_data),
            .pCode = vkcube_vert_data,
            .pNext = NULL
        },
        NULL,
        &vert_shader
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not load vertex shader. vkCreateShaderModule");
        return NULL;
    }

    ok = vkCreateShaderModule(
        dev->device,
        &(const VkShaderModuleCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .flags = 0,
            .codeSize = sizeof(vkcube_frag_data),
            .pCode = vkcube_frag_data,
            .pNext = NULL
        },
        NULL,
        &frag_shader
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not load vertex shader. vkCreateShaderModule");
        goto fail_destroy_vert_shader;
    }

    const VkPipelineShaderStageCreateInfo vert_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .flags = 0,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vert_shader,
        .pName = "main",
        .pSpecializationInfo = NULL,
        .pNext = NULL,
    };

    const VkPipelineShaderStageCreateInfo frag_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .flags = 0,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_shader,
        .pName = "main",
        .pSpecializationInfo = NULL,
        .pNext = NULL,
    };

    const VkPipelineShaderStageCreateInfo shader_stages[] = {
        vert_shader_stage_info,
        frag_shader_stage_info
    };

    // TODO: Make this static instead
    const VkPipelineDynamicStateCreateInfo dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .flags = 0,
        .dynamicStateCount = 2,
        .pDynamicStates = (VkDynamicState[2]) {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        },
        .pNext = NULL,
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .flags = 0,
        .vertexBindingDescriptionCount = 3,
        .pVertexBindingDescriptions = (VkVertexInputBindingDescription[]) {
            {
                .binding = 0,
                .stride = 3 * sizeof(float),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            },
            {
                .binding = 1,
                .stride = 3 * sizeof(float),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            },
            {
                .binding = 2,
                .stride = 3 * sizeof(float),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            },
        },
        .vertexAttributeDescriptionCount = 3,
        .pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]) {
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = 0,
            },
            {
                .location = 1,
                .binding = 1,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = 0,
            },
            {
                .location = 2,
                .binding = 2,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = 0,
            },
        },
        .pNext = NULL,
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = 0,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        .primitiveRestartEnable = VK_FALSE,
    };

    const VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = width,
        .height = height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const VkRect2D scissor = {
        .offset = {0, 0},
        .extent = {width, height}
    };

    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
        .pNext = NULL,
    };
    
    const VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
        .pNext = NULL,
    };

    const VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .pSampleMask = NULL,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
        .pNext = NULL,
    };

    const VkPipelineColorBlendAttachmentState color_blend_attachment = {
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
        .blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f },
        .pNext = NULL,
    };

    ok = vkCreateDescriptorSetLayout(
        dev->device,
        &(const VkDescriptorSetLayoutCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = (VkDescriptorSetLayoutBinding[]) {
                {
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                    .pImmutableSamplers = NULL
                }
            },
            .pNext = NULL,
        },
        NULL,
        &set_layout
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't create descriptor set layout. vkCreateDescriptorSetLayout");
        goto fail_destroy_frag_shader;
    }

    const VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = NULL,
        .pNext = NULL,
    };

    ok = vkCreatePipelineLayout(dev->device, &pipeline_layout_info, NULL, &pipeline_layout);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't create pipeline layout. vkCreatePipelineLayout");
        goto fail_destroy_descriptor_set_layout;
    }

    const VkAttachmentDescription color_attachment = {
        .flags = 0,
        .format = format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_GENERAL, // can be anything since we manually transition the image layout
    };

    const VkAttachmentReference color_attachment_reference = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    const VkSubpassDescription subpass = {
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = 0,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_reference,
        .pResolveAttachments = NULL,
        .pDepthStencilAttachment = NULL,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = NULL,
    };

    const VkRenderPassCreateInfo render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 0,
        .pDependencies = NULL,
        .pNext = NULL,
    };

    ok = vkCreateRenderPass(dev->device, &render_pass_info, NULL, &renderpass);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not create render pass. vkCreateRenderPass");
        goto fail_destroy_layout;
    }

    const VkGraphicsPipelineCreateInfo pipeline_create_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .flags = 0,
        .stageCount = 2,
        .pStages = shader_stages,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = NULL,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = NULL,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_state,
        .layout = pipeline_layout,
        .renderPass = renderpass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
        .pNext = NULL,
    };

    ok = vkCreateGraphicsPipelines(dev->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &vkpipeline);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't create graphics pipeline. vkCreateGraphicsPipeline");
        goto fail_destroy_layout;
    }

    pipeline = malloc(sizeof *pipeline);
    if (pipeline == NULL) {
        goto fail_destroy_pipeline;
    }

    pipeline->vert_shader = vert_shader;
    pipeline->frag_shader = frag_shader;
    pipeline->set_layout = set_layout;
    pipeline->pipeline_layout = pipeline_layout;
    pipeline->renderpass = renderpass;
    pipeline->pipeline = vkpipeline;
    return pipeline;

    
    fail_destroy_pipeline:
    vkDestroyPipeline(dev->device, vkpipeline, NULL);

    fail_destroy_renderpass:
    vkDestroyRenderPass(dev->device, renderpass, NULL);

    fail_destroy_layout:
    vkDestroyPipelineLayout(dev->device, pipeline_layout, NULL);

    fail_destroy_descriptor_set_layout:
    vkDestroyDescriptorSetLayout(dev->device, set_layout, NULL);

    fail_destroy_frag_shader:
    vkDestroyShaderModule(dev->device, frag_shader, NULL);

    fail_destroy_vert_shader:
    vkDestroyShaderModule(dev->device, vert_shader, NULL);
    return NULL;
}

void cube_pipeline_destroy(struct cube_pipeline *pipeline, VkDevice device) {
    vkDestroyPipeline(device, pipeline->pipeline, NULL);
    vkDestroyRenderPass(device, pipeline->renderpass, NULL);
    vkDestroyPipelineLayout(device, pipeline->pipeline_layout, NULL);
    vkDestroyShaderModule(device, pipeline->frag_shader, NULL);
    vkDestroyShaderModule(device, pipeline->vert_shader, NULL);
}

VkCommandBuffer cube_pipeline_record(struct vkdev *dev, struct cube_pipeline *pipeline, struct pipeline_fb *dest, struct cube_gpu_buffer *gpubuf) {
    VkCommandBuffer buffer;
    VkResult ok;

    ok = vkAllocateCommandBuffers(
        dev->device,
        &(const VkCommandBufferAllocateInfo) {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = dev->graphics_cmd_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
            .pNext = NULL,
        },
        &buffer
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not allocate command buffer for recording rendering commands. vkAllocateCommandBuffers");
        return VK_NULL_HANDLE;
    }

    ok = vkBeginCommandBuffer(
        buffer,
        &(const VkCommandBufferBeginInfo) {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
            .pInheritanceInfo = NULL,
            .pNext = NULL
        }
    );
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Could not begin recording rendering commands to command buffer. vkBeginCommandBuffer");
        return VK_NULL_HANDLE;
    }

    vkCmdBeginRenderPass(
        buffer,
        &(const VkRenderPassBeginInfo) {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = pipeline->renderpass,
            .framebuffer = dest->fb,
            .renderArea = {
                .offset = { 0.0f, 0.0f },
                .extent = {
                    dest->width,
                    dest->height,
                },
            },
            .clearValueCount = 1,
            .pClearValues = (const VkClearValue[1]) {
                {
                    .color = {{0.0f, 0.0f, 0.0f, 1.0f}},
                },
            },
            .pNext = NULL,
        },
        VK_SUBPASS_CONTENTS_INLINE
    );

    vkCmdBindVertexBuffers(
        buffer, 0, 3,
        (VkBuffer[]) {
            gpubuf->buffer,
            gpubuf->buffer,
            gpubuf->buffer
        },
        (VkDeviceSize[]) {
            offsetof(struct cube_gpu_data, vertices),
            offsetof(struct cube_gpu_data, colors),
            offsetof(struct cube_gpu_data, normals)
        }
    );

    vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);

    vkCmdBindDescriptorSets(
        buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline->pipeline_layout,
        0, 1,
        &gpubuf->descriptor_set,
        0,
        NULL
    );

    vkCmdSetViewport(
        buffer,
        0,
        1,
        &(const VkViewport) {
            .x = 0.0f,
            .y = 0.0f,
            .width = dest->width,
            .height = dest->height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        }
    );

    vkCmdSetScissor(
        buffer,
        0,
        1,
        &(const VkRect2D) {
            .offset = {0, 0},
            .extent = {
                dest->width,
                dest->height
            }
        }
    );

    vkCmdDraw(buffer, 4, 1,  0, 0);
    vkCmdDraw(buffer, 4, 1,  4, 0);
    vkCmdDraw(buffer, 4, 1,  8, 0);
    vkCmdDraw(buffer, 4, 1, 12, 0);
    vkCmdDraw(buffer, 4, 1, 16, 0);
    vkCmdDraw(buffer, 4, 1, 20, 0);

    vkCmdEndRenderPass(buffer);

    ok = vkEndCommandBuffer(buffer);
    if (ok != VK_SUCCESS) {
        LOG_VK_ERROR(ok, "Couldn't finish recording rendering commands. vkEndCommandBuffer");
        goto fail_free_buffer;
    }

    return buffer;


    fail_free_buffer:
    vkFreeCommandBuffers(dev->device, dev->graphics_cmd_pool, 1, &buffer);
    return VK_NULL_HANDLE;
}


struct vkkmscube {
    struct vkdev *vkdev;
    struct cube_pipeline *pipeline;

    int width, height;
    
    int drm_fd;
    struct drmdev *drmdev;
    struct gbm_device *gbm_device;

    struct {
        struct vk_kms_image *image;
        struct pipeline_fb *fb;
        VkCommandBuffer cmdbuf;
        uint32_t fb_id;
        struct cube_gpu_buffer *gpubuf;
    } images[4];
};

static VkBool32 on_debug_utils_message(
    VkDebugUtilsMessageSeverityFlagBitsEXT           severity,
    VkDebugUtilsMessageTypeFlagsEXT                  types,
    const VkDebugUtilsMessengerCallbackDataEXT*      data,
    void*                                            userdata
) {
    LOG_DEBUG(
        "[%s] (%"PRIi32", %s) %s (queues: %d, cmdbufs: %d, objects: %d)\n",
        severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT ? "VERBOSE" :
            severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT ? "INFO" :
            severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? "WARNING" :
            severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "ERROR" : "unknown severity",
        data->messageIdNumber,
        data->pMessageIdName,
        data->pMessage,
        data->queueLabelCount,
        data->cmdBufLabelCount,
        data->objectCount
    );
    return VK_TRUE;
}

static struct drmdev *create_and_configure_drmdev() {
    const struct drm_connector *connector;
    const struct drm_encoder *encoder;
    const struct drm_crtc *crtc;
    const drmModeModeInfo *mode, *mode_iter;
    struct drmdev *drmdev;
    drmDevicePtr devices[64];
    int n_devices;
    int ok;

    n_devices = drmGetDevices2(0, devices, sizeof(devices)/sizeof(*devices));
    if (n_devices < 0) {
        LOG_ERROR("Could not query DRM device list: %s\n", strerror(-n_devices));
        return NULL;
    }

    // find a GPU that has a primary node
    drmdev = NULL;
    for (int i = 0; i < n_devices; i++) {
        drmDevicePtr device;

        device = devices[i];

        if (!(device->available_nodes & (1 << DRM_NODE_PRIMARY))) {
            // We need a primary node.
            continue;
        }

        ok = drmdev_new_from_path(&drmdev, device->nodes[DRM_NODE_PRIMARY]);
        if (ok != 0) {
            LOG_ERROR("Could not create drmdev from device at \"%s\". Continuing.\n", device->nodes[DRM_NODE_PRIMARY]);
            continue;
        }

        break;
    }

    if (drmdev == NULL) {
        LOG_ERROR("Couldn't find a usable DRM device.\n"
                  "Please make sure you've enabled the Fake-KMS driver in raspi-config.\n"
                  "If you're not using a Raspberry Pi, please make sure there's KMS support for your graphics chip.\n");
        return NULL;
    }

    // find a connected connector
    for_each_connector_in_drmdev(drmdev, connector) {
        if (connector->connector->connection == DRM_MODE_CONNECTED) {
            break;
        }
    }

    if (connector == NULL) {
        LOG_ERROR("Could not find a connected connector!\n");
        return NULL;
    }

    // Find the preferred mode (GPU drivers _should_ always supply a preferred mode, but of course, they don't)
    // Alternatively, find the mode with the highest width*height. If there are multiple modes with the same w*h,
    // prefer higher refresh rates. After that, prefer progressive scanout modes.
    mode = NULL;
    for_each_mode_in_connector(connector, mode_iter) {
        if (mode_iter->type & DRM_MODE_TYPE_PREFERRED) {
            mode = mode_iter;
            break;
        } else if (mode == NULL) {
            mode = mode_iter;
        } else {
            int area = mode_iter->hdisplay * mode_iter->vdisplay;
            int old_area = mode->hdisplay * mode->vdisplay;

            if ((area > old_area) ||
                ((area == old_area) && (mode_iter->vrefresh > mode->vrefresh)) ||
                ((area == old_area) && (mode_iter->vrefresh == mode->vrefresh) && ((mode->flags & DRM_MODE_FLAG_INTERLACE) == 0))) {
                mode = mode_iter;
            }
        }
    }

    if (mode == NULL) {
        LOG_ERROR("Could not find a preferred output mode!\n");
        return NULL;
    }

    for_each_encoder_in_drmdev(drmdev, encoder) {
        if (encoder->encoder->encoder_id == connector->connector->encoder_id) {
            break;
        }
    }

    if (encoder == NULL) {
        for (int i = 0; i < connector->connector->count_encoders; i++, encoder = NULL) {
            for_each_encoder_in_drmdev(drmdev, encoder) {
                if (encoder->encoder->encoder_id == connector->connector->encoders[i]) {
                    break;
                }
            }

            if (encoder->encoder->possible_crtcs) {
                // only use this encoder if there's a crtc we can use with it
                break;
            }
        }
    }

    if (encoder == NULL) {
        LOG_ERROR("Could not find a suitable DRM encoder.\n");
        return NULL;
    }

    for_each_crtc_in_drmdev(drmdev, crtc) {
        if (crtc->crtc->crtc_id == encoder->encoder->crtc_id) {
            break;
        }
    }

    if (crtc == NULL) {
        for_each_crtc_in_drmdev(drmdev, crtc) {
            if (encoder->encoder->possible_crtcs & crtc->bitmask) {
                // find a CRTC that is possible to use with this encoder
                break;
            }
        }
    }

    if (crtc == NULL) {
        LOG_ERROR("Could not find a suitable DRM CRTC.\n");
        return NULL;
    }

    ok = drmdev_configure(drmdev, connector->connector->connector_id, encoder->encoder->encoder_id, crtc->crtc->crtc_id, mode);
    if (ok != 0) return NULL;

    return drmdev;
}

struct vkkmscube *vkkmscube_new() {
    struct cube_pipeline *cube_pipeline;
    struct gbm_device *gbm_device;
    struct vkkmscube *cube;
    struct drmdev *drmdev;
    struct vkdev *dev;
    int ok, drm_fd, width, height;

    static const VkFormat vk_format = VK_FORMAT_B8G8R8A8_SRGB;
    static const uint32_t drm_format = DRM_FORMAT_XRGB8888;
    static const uint32_t gbm_format = GBM_FORMAT_XRGB8888;

    cube = malloc(sizeof *cube);
    if (cube == NULL) {
        return NULL;
    }

    // clang-format off
    dev = vkdev_new(
        "vk-kmscube", VK_MAKE_VERSION(0, 0, 1),
        "vk-kmscube", VK_MAKE_VERSION(0, 0, 1),
        VK_MAKE_VERSION(1, 1, 0),
        (const char*[]) { "VK_LAYER_KHRONOS_validation", NULL }, NULL,
        (const char*[]) { "VK_EXT_debug_utils", NULL }, NULL,
        (const char*[]) {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
            VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
            VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,
            VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
            NULL
        },
        NULL,
        &(const struct debug_messenger) {
            .flags = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .severities = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .types = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .cb = on_debug_utils_message,
            .userdata = NULL
        }
    );
    // clang-format on
    if (dev == NULL) {
        LOG_ERROR("Could not setup vulkan device.\n");
        return NULL;
    }

    drmdev = create_and_configure_drmdev();
    if (drmdev == NULL) {
        LOG_ERROR("Couldn't open a KMS device\n");
        goto fail_destroy_vkdev;
    }

    drm_fd = drmdev->fd;
    width = drmdev->selected_mode->hdisplay;
    height = drmdev->selected_mode->vdisplay;

    cube_pipeline = cube_pipeline_new(dev, width, height, vk_format);
    if (cube_pipeline == NULL) {
        LOG_ERROR("Couldn't setup graphics pipeline.\n");
        goto fail_destroy_drmdev;
    }

    gbm_device = gbm_create_device(drm_fd);
    if (gbm_device == NULL) {
        LOG_ERROR("Couldn't create GBM device from KMS fd. gbm_create_device: %s\n", strerror(errno));
        goto fail_destroy_pipeline;
    }

    for (int i = 0; i < 4; i++) {
        struct vk_kms_image *img = vk_kms_image_new(dev, gbm_device, width, height, vk_format, gbm_format, drm_format, DRM_FORMAT_MOD_LINEAR);
        if (img == NULL) {
            LOG_ERROR("Couldn't create KMS image.\n");
            goto fail_destroy_previous;
        }

        uint32_t fb_id;
        ok = drmModeAddFB2WithModifiers(
            drm_fd,
            width, height,
            drm_format,
            (uint32_t[4]) { gbm_bo_get_handle_for_plane(img->bo, 0).u32, 0 },
            (uint32_t[4]) { gbm_bo_get_stride_for_plane(img->bo, 0), 0 },
            (uint32_t[4]) { gbm_bo_get_offset(img->bo, 0), 0 },
            (uint64_t[4]) { gbm_bo_get_modifier(img->bo), 0},
            &fb_id,
            0
        );
        if (ok < 0) {
            LOG_ERROR("Couldn't add GBM BO as kms image.\n");
            goto fail_destroy_pipeline_fb;
        }

        struct pipeline_fb *fb = pipeline_fb_new(dev, img, cube_pipeline->renderpass);
        if (fb == NULL) {
            LOG_ERROR("Couldn't import KMS FB into pipeline.\n");
            goto fail_destroy_kms_img;
        }

        struct cube_gpu_buffer *gpubuf = cube_gpu_buffer_new(dev, cube_pipeline->set_layout);
        if (gpubuf == NULL) {
            LOG_ERROR("Couldn't create a UBO/vertex buffer.\n");
            goto fail_destroy_pipeline_fb;
        }

        VkCommandBuffer cmdbuf = cube_pipeline_record(dev, cube_pipeline, fb, gpubuf);
        if (cmdbuf == VK_NULL_HANDLE) {
            LOG_ERROR("Couldn't record rendering commands.\n");
            goto fail_destroy_gpubuf;
        }
        
        cube->images[i].image = img;
        cube->images[i].fb = fb;
        cube->images[i].cmdbuf = cmdbuf;
        cube->images[i].fb_id = fb_id;
        cube->images[i].gpubuf = gpubuf;
        continue;


        fail_destroy_cmdbuf:
        vkFreeCommandBuffers(dev->device, dev->graphics_cmd_pool, 1, &cmdbuf);

        fail_destroy_gpubuf:
        cube_gpu_buffer_destroy(gpubuf, dev->device);

        fail_destroy_pipeline_fb:
        pipeline_fb_destroy(fb, dev->device);

        fail_rm_kms_fb:
        drmModeRmFB(drm_fd, fb_id);

        fail_destroy_kms_img:
        vk_kms_image_destroy(img, dev->device);

        fail_destroy_previous:
        for (int j = 0; j < i; j++) {
            cube_gpu_buffer_destroy(gpubuf, dev->device);
            vkFreeCommandBuffers(dev->device, dev->graphics_cmd_pool, 1, &(cube->images[j].cmdbuf));
            pipeline_fb_destroy(cube->images[j].fb, dev->device);
            drmModeRmFB(drm_fd, cube->images[j].fb_id);
            vk_kms_image_destroy(cube->images[j].image, dev->device);
        }
        goto fail_destroy_gbm_device;
    }

    cube->vkdev = dev;
    cube->pipeline = cube_pipeline;
    cube->drm_fd = drm_fd;
    cube->gbm_device = gbm_device;
    cube->drmdev = drmdev;
    cube->width = width;
    cube->height = height;
    return cube;


    fail_destroy_gbm_device:
    gbm_device_destroy(gbm_device);

    fail_destroy_pipeline:
    cube_pipeline_destroy(cube_pipeline, dev->device);

    fail_destroy_drmdev:
    close(drmdev->fd);

    fail_destroy_vkdev:
    vkdev_destroy(dev);

    fail_free_cube:
    free(cube);
    return NULL;
}

void vkkmscube_loop(struct vkkmscube *cube) {
    struct timeval start_time;
    VkResult vk_res;
    VkFence fence;
    int i, ok;

    vk_res = vkCreateFence(
        cube->vkdev->device,
        &(const VkFenceCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = 0,
            .pNext = NULL
        },
        NULL,
        &fence
    );
    if (vk_res != VK_SUCCESS) {
        LOG_VK_ERROR(vk_res, "Couldn't create fence to wait for rendering to complete. vkCreateFence");
        return;
    }

    gettimeofday(&start_time, NULL);

    LOG_DEBUG("looping\n");

    i = 0;
    while (1) {
        cube_gpu_buffer_update_transforms(cube->images[i].gpubuf, start_time, cube->height / (float) cube->width);

        vk_res = vkQueueSubmit(
            cube->vkdev->graphics_queue,
            1,
            &(const VkSubmitInfo) {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = NULL,
                .pWaitDstStageMask = NULL,
                .commandBufferCount = 1,
                .pCommandBuffers = &(cube->images[i].cmdbuf),
                .signalSemaphoreCount = 0,
                .pSignalSemaphores = NULL,
                .pNext = NULL,
            },
            fence
        );
        if (vk_res != VK_SUCCESS) {
            LOG_VK_ERROR(vk_res, "Couldn't submit command buffer. vkQueueSubmit");
            break;
        }

        vk_res = vkWaitForFences(cube->vkdev->device, 1, &fence, VK_TRUE, UINT64_MAX);
        if (vk_res != VK_SUCCESS) {
            LOG_VK_ERROR(vk_res, "Couldn't wait for rendering to complete. vkWaitForFences");
            break;
        }

        vk_res = vkResetFences(cube->vkdev->device, 1, &fence);
        if (vk_res != VK_SUCCESS) {
            LOG_VK_ERROR(vk_res, "Couldn't reset rendering fence. vkResetFences");
            break;
        }

        ok = drmModeSetCrtc(
            cube->drm_fd,
            cube->drmdev->selected_crtc->crtc->crtc_id,
            cube->images[i].fb_id,
            0, 0,
            &(cube->drmdev->selected_connector->connector->connector_id), 1,
            (drmModeModeInfoPtr) cube->drmdev->selected_mode
        );
        if (ok < 0) {
            LOG_ERROR("Couldn't set display mode. drmModeSetCrtc: %s\n", strerror(errno));
            return;
        }

        i = (i + 1) % 4;
    }
}

void vkkmscube_destroy(struct vkkmscube *cube) {
    LOG_DEBUG("destroying\n");
    cube_pipeline_destroy(cube->pipeline, cube->vkdev->device);
    vkdev_destroy(cube->vkdev);
    free(cube);
}


int main(int argc, char **argv) {
    struct vkkmscube *cube;

    cube = vkkmscube_new();
    if (cube == NULL) {
        return EXIT_FAILURE;
    }

    vkkmscube_loop(cube);

    vkkmscube_destroy(cube);

    LOG_DEBUG("Goodbye\n");
    return EXIT_SUCCESS;
}
