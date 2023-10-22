import numpy as np
import matplotlib.pyplot as plt
import os
import math;

# Open the file in binary mode
filename = 'Foot.byte'
volume_shape = (256, 256, 256)  ## cube root of filesize

with open(filename, 'rb') as byte_file:
    # Read the data into a NumPy array
    volume_data = np.fromfile(byte_file, dtype=np.uint8)

# Get the file size
file_size = os.path.getsize(filename)
# Reshape the 1D array into a 3D volume
volume_data = volume_data.reshape(volume_shape)

# Define camera parameters

image_plane_position = np.array([0, 0, -32])

# Create an image plane
width = 256;
height = 256;

rotate = 0;

initial_image_plane_center = np.array([width / 2, height / 2, -32])
object_plane_center = np.array([256 / 2, 256 / 2, 256 / 2])


def calculateTransformationCoordinate():
    finalPoint = np.array([0, 0, -32])
    return finalPoint - initial_image_plane_center


transformationCoordinate = calculateTransformationCoordinate()


def performInitialTransformation(x, y, z):
    InitialCoordinates = np.array([x, y, z])
    afterTransformation = InitialCoordinates + transformationCoordinate
    return afterTransformation


def performRotation(transformedArray):
    angle_radians = np.radians(rotate)
    rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                [0, 1, 0],
                                [-np.sin(angle_radians), 0, np.cos(angle_radians)]])

    afterRotation = np.dot(rotation_matrix, transformedArray)
    return afterRotation


def performFinalTransformation(rotatedArray):
    afterTransformation = rotatedArray + -(transformationCoordinate)
    return afterTransformation


def calculate_ray_direction():
    initialTransformation = performInitialTransformation(initial_image_plane_center[0], initial_image_plane_center[1],
                                                         initial_image_plane_center[2])
    afterRotation = performRotation(initialTransformation)
    final_image_plane_center = performFinalTransformation(afterRotation)
    ray_direction = object_plane_center - final_image_plane_center
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    return ray_direction


rd = calculate_ray_direction()


def calulate_intersection_point(x, y, z):
    r0 = np.array([x, y, z])
    pnAll = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    # rd = np.array([0,0,1])
    for pn in pnAll:
        vd = np.dot(pn, rd)
        D = np.array([0, -256])
        t = 0
        if (vd > 0):
            dot = np.dot(pn, r0)

            sums = -(dot + D)
            v0 = np.min(sums)
            t = v0 / vd
            xi = r0[0] + rd[0] * t
            yi = r0[1] + rd[1] * t
            zi = r0[2] + rd[2] * t
            return np.array([xi, yi, zi])


def getOpacity(val):

    opacity = val / 255
    return opacity


def getIntensity(val):

    intensity = val / 255
    return intensity


def interpolationCalc(intersectedArray):
    x = intersectedArray[0] - int(intersectedArray[0])
    y = intersectedArray[1] - int(intersectedArray[1])
    z = intersectedArray[2] - int(intersectedArray[2])
    getIntX = int(intersectedArray[0])
    getIntY = int(intersectedArray[1])
    getIntZ = int(intersectedArray[2])
    v1 = volume_data[getIntX, getIntY, getIntZ]
    v2 = volume_data[getIntX, getIntY, getIntZ + 1]
    v3 = volume_data[getIntX, getIntY + 1, getIntZ]
    v4 = volume_data[getIntX, getIntY + 1, getIntZ + 1]
    v5 = volume_data[getIntX + 1, getIntY, getIntZ]
    v6 = volume_data[getIntX + 1, getIntY, getIntZ + 1]
    v7 = volume_data[getIntX + 1, getIntY + 1, getIntZ]
    v8 = volume_data[getIntX + 1, getIntY + 1, getIntZ + 1]
    getValue = (1 - x) * (1 - y) * (1 - z) * v1 + (1 - x) * (1 - y) * (1 - z) * v2 + (1 - x) * (1 - y) * (
                1 - z) * v3 + (1 - x) * (1 - y) * (1 - z) * v4 + (1 - x) * (1 - y) * (1 - z) * v5 + (1 - x) * (
                           1 - y) * (1 - z) * v6 + (1 - x) * (1 - y) * (1 - z) * v7 + (1 - x) * (1 - y) * (1 - z) * v8
    print(getValue)
    return getValue


def volumeRayCasting(x, y):
    # transformation and rotation
    initialTransformation = performInitialTransformation(x, y, image_plane_position[2])

    rotation = performRotation(initialTransformation)
    # print('f', rotation)

    finalTransformation = performFinalTransformation(rotation)

    intersectionPoint = calulate_intersection_point(finalTransformation[0], finalTransformation[1],
                                                    finalTransformation[2])

    # intersectionPoint =  calulate_intersection_point(x, y, image_plane_position[2])
    # print(intersectionPoint)
    intensityFinal = 0
    accumulated_opacity = 1

    while intersectionPoint[0] < 255 and intersectionPoint[1] < 255 and intersectionPoint[2] < 255:
        # perform interpolation
        interpolatedVal = interpolationCalc(intersectionPoint)
        opacity = getOpacity(interpolatedVal)
        intensity = getIntensity(interpolatedVal)
        intensityFinal += intensity * accumulated_opacity
        accumulated_opacity *= (1 - opacity)
        if (accumulated_opacity == 0):
            break
        intersectionPoint = intersectionPoint + rd

    return intensityFinal


output_image = np.zeros((width, height, 3), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        intensity = volumeRayCasting(x, y)

        output_image[x, y, 0] = min(max(int(intensity * 255), 0), 255)  # Red channel
        output_image[x, y, 1] = min(max(int(intensity * 255), 0), 255)  # Green channel
        output_image[x, y, 2] = min(max(int(intensity * 255), 0), 255) # Blue channel

# print(output_image[253,253])
# print(volume_data[0,0,255])
# Display the rendered image
plt.imshow(output_image)
plt.axis('off')
plt.show()

