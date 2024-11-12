import cv2
import torch
import ctypes
import numpy as np
class SlicSuperPixel:
    def __init__(self, device = "cuda", iteration_N = 1, ruler=50, region_size=16, IsDrawPolyDecomp = False,  rgb_dist_threshold =10,
                 rgb_var_dist_threshold =10):
        self.device = device

        # SLIC params
        self.iteration_N = iteration_N
        self.ruler = ruler
        self.region_size = region_size

        # Convex decomposition
        self.lib = ctypes.CDLL('./scene/poly_decomp.dll')
        self.HertelMehlhornAPI = self.lib.HertelMehlhorn
        self.HertelMehlhornAPI.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                                           ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                           ctypes.c_bool]
        self.IsDrawPolyDecomp = IsDrawPolyDecomp

        # Color params for clustering superpixels
        self.rgb_dist_threshold = rgb_dist_threshold
        self.rgb_var_dist_threshold = rgb_var_dist_threshold

    def ComputeMergedSuperpixel(self, rgb):
        contour_img = rgb.copy()
        def dfs(graph, vertex, visited, component):
            visited.add(vertex)
            component.add(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    dfs(graph, neighbor, visited, component)


        slic = cv2.ximgproc.createSuperpixelSLIC(rgb, algorithm=102, region_size=self.region_size, ruler=self.ruler)
        slic.iterate(self.iteration_N)
        slic_labels = slic.getLabels()
        unique_labels = np.unique(slic_labels)

        # Get superpixel centers
        num_superpixels = slic.getNumberOfSuperpixels()
        centers = np.zeros((num_superpixels, 2), dtype=np.float32)

        # Contour of superpixels
        contour_list = []
        for i in range(num_superpixels):
            mask = np.uint8(slic_labels == i)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            center, _ = cv2.minEnclosingCircle(contours[0])
            centers[i] = center
            contour_list.append(contours)

        # Area of superpixels
        label_coordinates = []
        for label in unique_labels:
            indices = np.where(slic_labels == label)
            label_coordinates.append(list(zip(indices[1], indices[0])))  # (1-x, 0-y)

        # Compute neighboring superpixels
        bneighbors = set()

        rgb_lables = np.zeros((3, num_superpixels), dtype=np.float32)
        rgb_var_lables = np.zeros((3, num_superpixels), dtype=np.float32)
        num_of_pixels = np.zeros((1, num_superpixels), dtype=np.int32)

        for y in range(slic_labels.shape[0]):
            for x in range(slic_labels.shape[1]):
                current_label = slic_labels[y, x]
                num_of_pixels[0, current_label] += 1
                rgb_lables[:, current_label] += rgb[y, x]

        rgb_lables /= num_of_pixels

        for y in range(slic_labels.shape[0]):
            for x in range(slic_labels.shape[1]):
                current_label = slic_labels[y, x]
                rgb_var_lables[:, current_label] += np.sqrt((rgb[y, x] - rgb_lables[:, current_label]) ** 2)

        rgb_var_lables /= num_of_pixels

        for y in range(slic_labels.shape[0]):
            for x in range(slic_labels.shape[1]):
                current_label = slic_labels[y, x]
                if y > 0 and slic_labels[y - 1, x] != current_label:
                    bneighbors.add((current_label, slic_labels[y - 1, x]))
                if x > 0 and slic_labels[y, x - 1] != current_label:
                    bneighbors.add((current_label, slic_labels[y, x - 1]))

        # Create a plot to visualize the image with superpixel boundaries and centers
        for i in range(rgb.shape[0]):
            for j in range(rgb.shape[1]):
                if j + 1 < rgb.shape[1] and slic_labels[i, j] != slic_labels[i, j + 1]:
                    cv2.line(rgb, (j, i), (j + 1, i), (50, 50, 50), 1)
                if i + 1 < rgb.shape[0] and slic_labels[i, j] != slic_labels[i + 1, j]:
                    cv2.line(rgb, (j, i), (j, i + 1), (50, 50, 50), 1)

        # Cluster neighbors
        bneighbors_copy = bneighbors.copy()
        for neighbor_pair in bneighbors_copy:
            label1, label2 = neighbor_pair
            rgb_mean1 = rgb_lables[:, label1]
            rgb_mean2 = rgb_lables[:, label2]
            rgb_dist = np.sqrt(np.sum((rgb_mean1 - rgb_mean2) ** 2))
            rgb_var1 = rgb_var_lables[:, label1]
            rgb_var2 = rgb_var_lables[:, label2]
            rgb_var_dist = np.sqrt(np.sum((rgb_var1 - rgb_var2) ** 2))
            if rgb_dist > self.rgb_dist_threshold or \
                    rgb_var_dist > self.rgb_var_dist_threshold:
                bneighbors.remove(neighbor_pair)

        graph = {}
        for edge in bneighbors:
            u, v = edge
            if u not in graph:
                graph[u] = set()
            if v not in graph:
                graph[v] = set()
            graph[u].add(v)
            graph[v].add(u)

        visited = set()
        connected_components = []
        for vertex in graph:
            if vertex not in visited:
                component = set()
                dfs(graph, vertex, visited, component)
                connected_components.append(component)

        connected_components_edges = []
        for component in connected_components:
            component_edges = set()
            cluster = []
            for vertex in component:
                if vertex not in cluster:
                    cluster.append(vertex)
                for neighbor in graph[vertex]:
                    if (vertex, neighbor) in bneighbors:
                        component_edges.add((vertex, neighbor))
                        if neighbor not in cluster:
                            cluster.append(neighbor)
            connected_components_edges.append(cluster.copy())

        # Draw superpixel centers
        color_list = [(100, 100, 100), (100, 100, 255), (100, 255, 100), (100, 255, 255),
                      (255, 100, 100), (255, 100, 255), (255, 255, 100), (200, 200, 200)]

        for center in centers:
            cv2.circle(rgb, (int(center[0]), int(center[1])), 2, (0, 0, 255), -1)


        label_candidate = list(range(num_superpixels))
        polygon_center_list = np.empty(shape=(0, 2), dtype=np.int32)
        # print("sp", range(num_superpixels))

        contour_flag = np.full((len(label_coordinates), 1), True)
        contour_edge = contour_img.copy()
        filter_contour_flag = np.full((len(label_coordinates), 1), True)
        pl_mid = 0
        for i, cluster in enumerate(connected_components_edges):
            mask_image = np.zeros((contour_edge.shape[0], contour_edge.shape[1], 1), dtype=np.uint8)
            color = color_list[i%8]
            cluster_mask = []
            cluster_size = len(cluster)
            for label in cluster:
                if contour_flag[label]:
                    cluster_mask += label_coordinates[label]
                    contour_flag[label] = False

            cluster_mask_np = np.array(cluster_mask).reshape((-1, 1, 2))
            cv2.drawContours(contour_img, cluster_mask_np, -1, color, 1)
            cv2.drawContours(mask_image, cluster_mask_np, -1, (255), 1)

            contours, _ = cv2.findContours(mask_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


            poly_list = []
            hole_exist = False
            for j, contour in enumerate(contours):
                contour = np.array(contour)
                if len(contour.shape) == 3 and contour.shape[0] == 1:
                    contour = contour[0]
                epsilon = 0.008 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                # cv2.polylines(contour_edge, [approx], isClosed=True, color=color, thickness=2)
                approx = np.array(approx, dtype=np.float32).reshape((-1, 2))
                poly_list.append(approx)
                if not hole_exist:
                    break
            mid_list, polygon_list = self.HertelMehlhorn(poly_list, self.IsDrawPolyDecomp)
            pl_mid+= len(mid_list)

            # Decomposed covex shapes are selected,
            # if number of them is smaller than original superpixels
            # print("cl: ", cluster_size, "poly_size: ", len(polygon_list))
            if cluster_size > len(mid_list):
                for label in cluster:
                    if filter_contour_flag[label]:
                        filter_contour_flag[label] = False
                    label_candidate.remove(label) # candidate에서 label삭제
                polygon_center_list = np.concatenate((polygon_center_list, mid_list), axis=0)

        result_midpoints_list = []
        for i, flag in enumerate(filter_contour_flag):
            if flag:
                center = centers[i]
                result_midpoints_list.append((int(center[1]), int(center[0])))

        # print("result", len(result_midpoints_list))

        result_midpoints_np = (np.array(result_midpoints_list)).reshape((-1, 2))
        result_midpoints_np = np.concatenate((result_midpoints_np, polygon_center_list), axis=0)
        result_midpoints = torch.from_numpy(result_midpoints_np).to(self.device).to(torch.int).T
        return result_midpoints

        # Display the image
        cv2.imshow("contor", contour_img)
        cv2.imshow("contour_edge", contour_edge)
        cv2.imshow("Superpixels", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def HertelMehlhorn(self, poly_list, is_polygon_list):
        # Convert the list of NumPy arrays to a ctypes array of pointers
        poly_list_ptrs = (ctypes.POINTER(ctypes.c_float) * len(poly_list))()
        poly_len_list_ptrs = (ctypes.c_int * len(poly_list))()
        for i, poly in enumerate(poly_list):
            poly_list_ptrs[i] = poly.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            poly_len_list_ptrs[i] = poly.shape[0]

        size_mid_list = ctypes.c_int()
        mid_list = ctypes.POINTER(ctypes.c_int)()
        polygon_size_list = ctypes.POINTER(ctypes.c_int)()
        polygon_vertices = ctypes.POINTER(ctypes.c_float)()

        # Call the C++ function
        self.HertelMehlhornAPI(poly_list_ptrs, len(poly_list), poly_len_list_ptrs,
                        ctypes.byref(size_mid_list), ctypes.byref(mid_list),
                        ctypes.byref(polygon_size_list), ctypes.byref(polygon_vertices),
                        is_polygon_list)

        coordinates_array = np.ctypeslib.as_array(mid_list, shape=(size_mid_list.value, 2))

        polygon_list = []
        if is_polygon_list:
            polygon_size_list_result = np.ctypeslib.as_array(polygon_size_list, shape=(size_mid_list.value, 1))
            vertex_size = 0
            for poly_size in polygon_size_list_result:
                vertex_size += poly_size[0]
            polygon_list_result = np.ctypeslib.as_array(polygon_vertices, shape=(vertex_size, 2))
            vertex_size = 0
            for poly_size in polygon_size_list_result:
                polygon = []
                for i in range(vertex_size, vertex_size + poly_size[0]):
                    polygon.append(polygon_list_result[i])
                polygon_list.append(polygon.copy())
                vertex_size += poly_size[0]

        return coordinates_array, polygon_list
