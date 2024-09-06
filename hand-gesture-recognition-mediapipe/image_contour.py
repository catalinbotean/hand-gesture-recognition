import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from functools import reduce

# Constants
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


# Function to calculate the length of a single contour
def contour_length_single(contour):
    lengths = [np.linalg.norm(np.array(contour[i]) - np.array(contour[i - 1])) for i in range(1, len(contour))]
    return sum(lengths)


# Function to calculate the length of all contours
def contour_length(contours):
    with ProcessPoolExecutor() as executor:
        lengths = list(executor.map(contour_length_single, contours))
    return lengths


# Function to extract subpixel facet
def sub_pixel_facet(p, gyMat, gxMat, gyyMat, gxxMat, gxyMat):
    row, col = p.y, p.x
    gy = gyMat[row, col]
    gx = gxMat[row, col]
    gyy = gyyMat[row, col]
    gxx = gxxMat[row, col]
    gxy = gxyMat[row, col]

    hessian = np.array([[gyy, gxy], [gxy, gxx]])
    _, v = np.linalg.eigh(hessian)
    ny, nx = v[:, 0]
    t = -(gx * nx + gy * ny) / (gxx * nx * nx + 2 * gxy * nx * ny + gyy * ny * ny)
    px = t * nx
    py = t * ny

    return (col + px, row + py)


# Function to calculate subpixel points for a single contour
def sub_pixel_single(gy, gx, gyy, gxx, gxy, cont):
    return [sub_pixel_facet(p, gy, gx, gyy, gxx, gxy) for p in cont]


# Function to process subpixel edge contour
def sub_pixel_edge_contour(image_gray, filteredCont):
    p_vec = np.array([0.004711, 0.069321, 0.245410, 0.361117, 0.245410, 0.069321, 0.004711])
    d1_vec = np.array([-0.018708, -0.125376, -0.193091, 0.000000, 0.193091, 0.125376, 0.018708])
    d2_vec = np.array([0.055336, 0.137778, -0.056554, -0.273118, -0.056554, 0.137778, 0.055336])

    dy = cv2.sepFilter2D(image_gray, cv2.CV_64F, p_vec, d1_vec)
    dx = cv2.sepFilter2D(image_gray, cv2.CV_64F, d1_vec, p_vec)
    grad = np.sqrt(dy * dy + dx * dx)

    gy = cv2.sepFilter2D(grad, cv2.CV_64F, p_vec, d1_vec)
    gx = cv2.sepFilter2D(grad, cv2.CV_64F, d1_vec, p_vec)
    gyy = cv2.sepFilter2D(grad, cv2.CV_64F, p_vec, d2_vec)
    gxx = cv2.sepFilter2D(grad, cv2.CV_64F, d2_vec, p_vec)
    gxy = cv2.sepFilter2D(grad, cv2.CV_64F, d1_vec, d1_vec)

    with ProcessPoolExecutor() as executor:
        cont_sub_pix_full = list(executor.map(lambda cont: sub_pixel_single(gy, gx, gyy, gxx, gxy, cont), filteredCont))

    return cont_sub_pix_full


# Function to get valid contour indices
def get_edge_contour_valid_indices(hierarchy):
    NEXT_SAME = 0
    FIRST_CHILD = 2

    valid_indices = []
    exclude_indices = []
    index = 0
    while index != -1:
        if hierarchy[index][FIRST_CHILD] != -1:
            exclude_indices.append(index)
        index = hierarchy[index][NEXT_SAME]

    set_full_indices = set(range(len(hierarchy)))
    valid_indices = list(set_full_indices - set(exclude_indices))

    return valid_indices, exclude_indices


def main():
    image_path = "./"
    output_image_path = "./img_out"
    os.makedirs(output_image_path, exist_ok=True)
    filename = "/Users/catalinbotean/Desktop/lion.png"

    image = cv2.imread(os.path.join(image_path, filename), cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"{image.shape[0]},{image.shape[1]}")

    edgeIm = cv2.Canny(image_gray, 180, 200)
    cv2.imwrite(os.path.join(output_image_path, "outputEdge.png"), edgeIm)

    contours, hierarchy = cv2.findContours(edgeIm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    valid_indices, exclude_indices = get_edge_contour_valid_indices(hierarchy)

    inner_contours = [contours[i] for i in valid_indices]

    contourIm = np.zeros_like(image)
    print(len(inner_contours))
    for i, contour in enumerate(inner_contours):
        cv2.drawContours(contourIm, inner_contours, i, COLORS[i % len(COLORS)], 1, cv2.LINE_8)

    cv2.imwrite(os.path.join(output_image_path, "innerContours.png"), contourIm)

    external_contours = [contours[i] for i in exclude_indices]

    contourExtIm = np.zeros_like(image)
    print(len(external_contours))
    for i, contour in enumerate(external_contours):
        cv2.drawContours(contourExtIm, external_contours, i, COLORS[i % len(COLORS)], 1, cv2.LINE_8)

    cv2.imwrite(os.path.join(output_image_path, "externalContours.png"), contourExtIm)

    cont_lengths = contour_length(inner_contours)
    cont_length_mat = np.array(cont_lengths)

    cont_ra = []
    cont_aspect_ratio = []
    cont_center = []
    for c in inner_contours:
        M = cv2.moments(c)

        area = M['m00']
        centerX = int(M['m10'] / area)
        centerY = int(M['m01'] / area)
        m20 = M['mu20'] / area
        m02 = M['mu02'] / area
        m11 = M['mu11'] / area
        c1 = m20 - m02
        c2 = c1 ** 2
        c3 = 4 * m11 ** 2

        cont_center.append((centerX, centerY))

        ra = np.sqrt(2.0 * (m20 + m02 + np.sqrt(c2 + c3)))
        rb = np.sqrt(2.0 * (m20 + m02 - np.sqrt(c2 + c3)))
        cont_ra.append(ra)
        cont_aspect_ratio.append(ra / rb)

    cont_radius_mat = np.array(cont_ra)
    cont_aspect_ratio_mat = np.array(cont_aspect_ratio)

    print(f"{cont_radius_mat.shape[0]},{cont_radius_mat.shape[1]}")
    RADIUS_MAX = 5
    CONT_LENGTH_MAX = 2 * np.pi * RADIUS_MAX
    _, thres_aspect_ratio = cv2.threshold(cont_aspect_ratio_mat, 0.8, 1.0, cv2.THRESH_BINARY)
    _, thres_radius = cv2.threshold(cont_radius_mat, RADIUS_MAX, 1.0, cv2.THRESH_BINARY_INV)
    _, thres_cont_length = cv2.threshold(cont_length_mat, CONT_LENGTH_MAX, 1.0, cv2.THRESH_BINARY_INV)

    and1 = cv2.bitwise_and(thres_aspect_ratio, thres_radius)
    and2 = cv2.bitwise_and(and1, thres_cont_length)
    print(f"Filtered object count: {int(np.sum(and2))}")

    filtered_idx = np.nonzero(and2)[0]

    filtered_cont = [inner_contours[i] for i in filtered_idx]
    filtered_cont_center = [cont_center[i] for i in filtered_idx]

    cont_sub_pix_full = sub_pixel_edge_contour(image_gray, filtered_cont)

    with open("./data.txt", "w") as f:
        f.write("contour_id,point_id,x,y\n")
        for i, sub_pix in enumerate(cont_sub_pix_full):
            for j, p in enumerate(sub_pix):
                line = f"{i},{j},{p[0]:.4f},{p[1]:.4f}\n"
                f.write(line)

    crop_half_width = 7
    up_scale_factor = 50
    final_result_count = 4

    for result_index in range(final_result_count):
        x_crop_start = filtered_cont_center[result_index][0] - crop_half_width
        y_crop_start = filtered_cont_center[result_index][1] - crop_half_width
        rect = (x_crop_start, y_crop_start, 2 * crop_half_width + 1, 2 * crop_half_width + 1)

        crop = image[y_crop_start:y_crop_start + rect[3], x_crop_start:x_crop_start + rect[2]]

        up_scaled_width = up_scale_factor * (2 * crop_half_width + 1)
        up_scaled = np.zeros((up_scaled_width, up_scaled_width, 3), dtype=np.uint8)

        for i in range(crop.shape[1]):
            for j in range(crop.shape[0]):
                up_scaled[j * up_scale_factor:(j + 1) * up_scale_factor,
                i * up_scale_factor:(i + 1) * up_scale_factor] = crop[j, i]

        display_contour = []
        for p in cont_sub_pix_full[result_index]:
            x = int(((p[0] - x_crop_start) + 0.5) * up_scale_factor)
            y = int(((p[1] - y_crop_start) + 0.5) * up_scale_factor)
            cv2.drawMarker(up_scaled, (x, y), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 3)
            display_contour.append((x, y))

        display_contour_full = [display_contour]
        cv2.drawContours(up_scaled, display_contour_full, 0, (255, 0, 0), 3)

        cv2.imwrite(os.path.join(output_image_path, f"final-{result_index:02d}.png"), up_scaled)


if __name__ == "__main__":
    main()
