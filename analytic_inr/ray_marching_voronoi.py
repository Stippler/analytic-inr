"""
Ray Marching Algorithm for Edge-Based Voronoi Diagrams in 2D/3D

This module implements an analytical approach to computing Voronoi diagrams where
each polygon edge (or triangle in 3D) gets its own Voronoi cell.

Algorithm:
1. Start from each vertex with a normal direction (average of incident edge normals)
2. March rays step-by-step from each vertex
3. Check for ray-ray collisions within a circular/spherical radius
4. Stop marching when rays intersect
5. Build graph structure from intersection points
6. Optionally perform secondary marching for complete diagram

This is fully parallelized using PyTorch for efficient computation.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class RayMarchingVoronoi:
    """
    Compute edge-based Voronoi diagrams using ray marching in 2D.
    """
    
    def __init__(self, polygons: List[torch.Tensor], device='cpu'):
        """
        Initialize with a list of polygons.
        
        Args:
            polygons: List of torch.Tensors, each of shape (n, 2) with vertex coordinates
            device: Device to run computation on ('cpu' or 'cuda')
        """
        self.device = device
        self.polygons = [p.to(device) for p in polygons]
        
        # Extract vertices and edges
        self.vertices, self.edges, self.vertex_to_edges = self._extract_graph_structure()
        
        # Results
        self.voronoi_vertices = None
        self.voronoi_edges = None
        self.collision_data = None
        
    def _extract_graph_structure(self):
        """
        Extract vertices, edges, and connectivity from polygons.
        
        Returns:
            vertices: torch.Tensor of shape (num_vertices, 2)
            edges: torch.Tensor of shape (num_edges, 2, 2) - [start, end] pairs
            vertex_to_edges: Dict mapping vertex index to list of incident edge indices
        """
        all_vertices = []
        all_edges = []
        vertex_to_edges = {}
        
        vertex_offset = 0
        edge_idx = 0
        
        for polygon in self.polygons:
            num_verts = polygon.shape[0]
            
            # Add vertices
            all_vertices.append(polygon)
            
            # Add edges and build connectivity
            for i in range(num_verts):
                v1_global = vertex_offset + i
                v2_global = vertex_offset + (i + 1) % num_verts
                
                start = polygon[i]
                end = polygon[(i + 1) % num_verts]
                all_edges.append(torch.stack([start, end]))
                
                # Track which edges are incident to which vertices
                if v1_global not in vertex_to_edges:
                    vertex_to_edges[v1_global] = []
                if v2_global not in vertex_to_edges:
                    vertex_to_edges[v2_global] = []
                
                vertex_to_edges[v1_global].append(edge_idx)
                vertex_to_edges[v2_global].append(edge_idx)
                
                edge_idx += 1
            
            vertex_offset += num_verts
        
        vertices = torch.cat(all_vertices, dim=0)
        edges = torch.stack(all_edges, dim=0)
        
        return vertices, edges, vertex_to_edges
    
    def compute_vertex_normals(self):
        """
        Compute outward-pointing normal at each vertex for CCW-ordered polygons.
        
        For each vertex, we find the two incident edges (incoming and outgoing),
        compute the outward normal of each edge, and average them.
        
        For CCW polygons, the outward normal of an edge with direction (dx, dy)
        is obtained by 90° counter-clockwise rotation: (-dy, dx)
        
        Returns:
            torch.Tensor of shape (num_vertices, 2) with normalized normals
        """
        num_vertices = self.vertices.shape[0]
        normals = torch.zeros(num_vertices, 2, device=self.device)
        
        vertex_offset = 0
        
        # Process each polygon separately to maintain CCW ordering
        for polygon in self.polygons:
            num_verts = polygon.shape[0]
            
            for i in range(num_verts):
                v_idx = vertex_offset + i
                current_vertex = polygon[i]
                
                # Get previous and next vertices (CCW order)
                prev_vertex = polygon[(i - 1) % num_verts]
                next_vertex = polygon[(i + 1) % num_verts]
                
                # Incoming edge: prev -> current
                incoming_dir = current_vertex - prev_vertex
                # Outgoing edge: current -> next
                outgoing_dir = next_vertex - current_vertex
                
                # Compute outward normals (90° CCW rotation: (dx, dy) -> (-dy, dx))
                incoming_normal = torch.stack([-incoming_dir[1], incoming_dir[0]])
                outgoing_normal = torch.stack([-outgoing_dir[1], outgoing_dir[0]])
                
                # Normalize each
                incoming_normal = incoming_normal / (torch.norm(incoming_normal) + 1e-10)
                outgoing_normal = outgoing_normal / (torch.norm(outgoing_normal) + 1e-10)
                
                # Average and normalize
                avg_normal = (incoming_normal + outgoing_normal) / 2.0
                normals[v_idx] = avg_normal / (torch.norm(avg_normal) + 1e-10)
            
            vertex_offset += num_verts
        
        return normals
    
    def compute_ray_intersection_2d(self, pos1, dir1, pos2, dir2):
        """
        Compute intersection point of two 2D rays.
        
        Args:
            pos1, pos2: Ray origins, shape (N, 2) or (2,)
            dir1, dir2: Ray directions, shape (N, 2) or (2,)
        
        Returns:
            intersections: torch.Tensor of shape (N, 2) or (2,)
            valid: torch.Tensor of shape (N,) or scalar - True if valid intersection
        """
        if pos1.dim() == 1:
            pos1 = pos1.unsqueeze(0)
            dir1 = dir1.unsqueeze(0)
            pos2 = pos2.unsqueeze(0)
            dir2 = dir2.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        # Solve: pos1 + t1*dir1 = pos2 + t2*dir2
        # Rearrange: t1*dir1 - t2*dir2 = pos2 - pos1
        # In matrix form: [dir1 | -dir2] * [t1; t2] = delta
        
        delta = pos2 - pos1  # (N, 2)
        
        # Use cross product for 2D intersection
        cross = dir1[:, 0] * dir2[:, 1] - dir1[:, 1] * dir2[:, 0]  # (N,)
        
        # Check if rays are parallel
        valid = torch.abs(cross) > 1e-8
        
        t1 = (delta[:, 0] * dir2[:, 1] - delta[:, 1] * dir2[:, 0]) / (cross + 1e-10)
        t2 = (delta[:, 0] * dir1[:, 1] - delta[:, 1] * dir1[:, 0]) / (cross + 1e-10)
        
        # Both t values should be positive for valid intersection
        valid = valid & (t1 > 0) & (t2 > 0)
        
        intersections = pos1 + t1.unsqueeze(1) * dir1
        
        if squeeze:
            return intersections.squeeze(0), valid.squeeze(0)
        return intersections, valid
    
    def ray_intersects_segment(self, ray_pos, ray_dir, seg_start, seg_end):
        """
        Check if a ray intersects with a line segment.
        
        Args:
            ray_pos: Ray origin, shape (2,)
            ray_dir: Ray direction, shape (2,)
            seg_start: Segment start, shape (2,)
            seg_end: Segment end, shape (2,)
        
        Returns:
            intersection: torch.Tensor of shape (2,) - intersection point
            valid: bool - True if valid intersection exists
            t_ray: float - parameter along ray (should be > 0)
            t_seg: float - parameter along segment (should be in [0, 1])
        """
        seg_dir = seg_end - seg_start
        delta = seg_start - ray_pos
        
        # Solve: ray_pos + t_ray*ray_dir = seg_start + t_seg*seg_dir
        cross = ray_dir[0] * seg_dir[1] - ray_dir[1] * seg_dir[0]
        
        if torch.abs(cross) < 1e-8:
            # Parallel
            return torch.zeros(2, device=self.device), False, 0.0, 0.0
        
        t_ray = (delta[0] * seg_dir[1] - delta[1] * seg_dir[0]) / cross
        t_seg = (delta[0] * ray_dir[1] - delta[1] * ray_dir[0]) / cross
        
        valid = (t_ray > 1e-6) and (0 <= t_seg <= 1)
        
        intersection = ray_pos + t_ray * ray_dir
        
        return intersection, valid, t_ray.item(), t_seg.item()
    
    def compute_boundary_intersection(self, point, direction, bounds=(-1, 1)):
        """
        Compute where a line through point with given direction intersects domain boundary.
        
        Args:
            point: Point on line, shape (2,)
            direction: Line direction (unit vector), shape (2,)
            bounds: Domain bounds (min, max)
        
        Returns:
            t_min, t_max: Parameters for the two boundary intersections
        """
        # Line: p = point + t * direction
        # Find t values where line crosses boundary
        
        t_values = []
        
        # Check x boundaries
        if abs(direction[0]) > 1e-8:
            t_xmin = (bounds[0] - point[0]) / direction[0]
            t_xmax = (bounds[1] - point[0]) / direction[0]
            t_values.extend([t_xmin, t_xmax])
        
        # Check y boundaries
        if abs(direction[1]) > 1e-8:
            t_ymin = (bounds[0] - point[1]) / direction[1]
            t_ymax = (bounds[1] - point[1]) / direction[1]
            t_values.extend([t_ymin, t_ymax])
        
        # Filter to valid range and sort
        valid_t = []
        for t in t_values:
            p = point + t * direction
            if (bounds[0] <= p[0] <= bounds[1] and 
                bounds[0] <= p[1] <= bounds[1]):
                valid_t.append(t.item())
        
        valid_t = sorted(valid_t)
        
        if len(valid_t) >= 2:
            return valid_t[0], valid_t[-1]
        else:
            return -1000.0, 1000.0  # Fallback
    
    def create_boundary_segments(self, bounds=(-1, 1)):
        """
        Create line segments from each vertex along its normal direction.
        Segments extend from boundary to boundary through the vertex.
        
        Args:
            bounds: Domain bounds (min, max), default (-1, 1)
        
        Returns:
            dict with:
                - segments: List of (start_point, end_point) for each vertex
                - vertex_indices: Which vertex each segment corresponds to
        """
        normals = self.compute_vertex_normals()
        num_vertices = self.vertices.shape[0]
        
        segments = []
        vertex_indices = []
        
        print(f"\nCreating boundary-to-boundary segments...")
        print(f"  Domain: [{bounds[0]}, {bounds[1]}]^2")
        
        for v_idx in range(num_vertices):
            vertex = self.vertices[v_idx]
            normal = normals[v_idx]
            
            # Find where line intersects boundaries
            t_min, t_max = self.compute_boundary_intersection(vertex, normal, bounds)
            
            # Create segment endpoints
            start_point = vertex + t_min * normal
            end_point = vertex + t_max * normal
            
            segments.append((start_point, end_point))
            vertex_indices.append(v_idx)
        
        print(f"  Created {len(segments)} segments")
        
        return {
            'segments': segments,
            'vertex_indices': vertex_indices,
            'normals': normals
        }
    
    def find_segment_intersections(self, segments_data):
        """
        Find intersections between line segments.
        For each vertex, find the NEAREST intersection in BOTH directions along its normal.
        
        Algorithm:
        - For each vertex, we have a line through it along its normal
        - Find the nearest intersection in the POSITIVE direction (normal direction)
        - Find the nearest intersection in the NEGATIVE direction (-normal direction)
        - These are the Voronoi vertices for this vertex's bisector
        
        Args:
            segments_data: Output from create_boundary_segments()
        
        Returns:
            dict with:
                - voronoi_vertices: List of Voronoi vertices (intersection points)
                - vertex_to_voronoi: Mapping from vertex to its Voronoi vertices (pos/neg)
        """
        segments = segments_data['segments']
        vertex_indices = segments_data['vertex_indices']
        normals = segments_data['normals']
        num_segments = len(segments)
        
        print(f"\nFinding segment intersections (nearest in each direction)...")
        
        # For each vertex, find nearest intersection in both directions
        voronoi_vertices = []
        voronoi_pairs = []
        vertex_to_voronoi = {}
        
        for i in range(num_segments):
            seg1_start, seg1_end = segments[i]
            v1_idx = vertex_indices[i]
            vertex1 = self.vertices[v1_idx]
            normal1 = normals[v1_idx]
            
            # Find nearest intersection in positive direction
            nearest_pos = None
            nearest_pos_dist = float('inf')
            nearest_pos_partner = None
            
            # Find nearest intersection in negative direction
            nearest_neg = None
            nearest_neg_dist = float('inf')
            nearest_neg_partner = None
            
            for j in range(num_segments):
                if i == j:
                    continue
                
                seg2_start, seg2_end = segments[j]
                v2_idx = vertex_indices[j]
                
                # Find intersection
                int_point, valid = self.line_segment_intersection(
                    seg1_start, seg1_end, seg2_start, seg2_end
                )
                
                if valid:
                    # Determine which direction this intersection is in
                    to_intersection = int_point - vertex1
                    dist = torch.norm(to_intersection).item()
                    
                    # Project onto normal to determine direction
                    projection = torch.dot(to_intersection, normal1).item()
                    
                    if projection > 1e-6:  # Positive direction
                        if dist < nearest_pos_dist:
                            nearest_pos_dist = dist
                            nearest_pos = int_point
                            nearest_pos_partner = v2_idx
                    elif projection < -1e-6:  # Negative direction
                        if dist < nearest_neg_dist:
                            nearest_neg_dist = dist
                            nearest_neg = int_point
                            nearest_neg_partner = v2_idx
            
            # Store results for this vertex
            vertex_to_voronoi[v1_idx] = {}
            
            if nearest_pos is not None:
                voronoi_vertices.append(nearest_pos)
                voronoi_pairs.append((v1_idx, nearest_pos_partner))
                vertex_to_voronoi[v1_idx]['positive'] = {
                    'point': nearest_pos,
                    'distance': nearest_pos_dist,
                    'partner': nearest_pos_partner,
                    'index': len(voronoi_vertices) - 1
                }
            
            if nearest_neg is not None:
                voronoi_vertices.append(nearest_neg)
                voronoi_pairs.append((v1_idx, nearest_neg_partner))
                vertex_to_voronoi[v1_idx]['negative'] = {
                    'point': nearest_neg,
                    'distance': nearest_neg_dist,
                    'partner': nearest_neg_partner,
                    'index': len(voronoi_vertices) - 1
                }
        
        print(f"  Found {len(voronoi_vertices)} Voronoi vertices")
        print(f"  ({len(voronoi_vertices)//2} vertices × 2 directions)")
        
        return {
            'voronoi_vertices': voronoi_vertices,
            'voronoi_pairs': voronoi_pairs,
            'vertex_to_voronoi': vertex_to_voronoi
        }
    
    def line_segment_intersection(self, p1, p2, p3, p4):
        """
        Find intersection of two line segments.
        Segment 1: p1 to p2
        Segment 2: p3 to p4
        
        Returns:
            intersection_point, is_valid
        """
        d1 = p2 - p1
        d2 = p4 - p3
        delta = p3 - p1
        
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        
        if abs(cross) < 1e-8:
            # Parallel or collinear
            return torch.zeros(2, device=self.device), False
        
        t1 = (delta[0] * d2[1] - delta[1] * d2[0]) / cross
        t2 = (delta[0] * d1[1] - delta[1] * d1[0]) / cross
        
        # Check if intersection is within both segments
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            intersection = p1 + t1 * d1
            return intersection, True
        
        return torch.zeros(2, device=self.device), False
    
    def march_rays(self, step_size=0.01, max_steps=500, collision_radius=None):
        """
        Perform ray marching from all vertices in parallel.
        Now checks for ray-segment intersections (not just point-point proximity).
        
        Args:
            step_size: Distance to march per step (default: 0.01)
            max_steps: Maximum number of steps (default: 500)
            collision_radius: Not used anymore (kept for compatibility)
        
        Returns:
            dict with:
                - voronoi_vertices: Intersection points
                - collision_pairs: Which original vertices created each intersection
                - ray_trajectories: Full trajectories for visualization
        """
        num_vertices = self.vertices.shape[0]
        
        # Initialize rays
        ray_origins = self.vertices.clone()
        ray_directions = self.compute_vertex_normals()
        ray_positions = ray_origins.clone()
        ray_active = torch.ones(num_vertices, dtype=torch.bool, device=self.device)
        
        # Storage for results
        collision_points = []
        collision_pairs = []
        collision_steps = []  # Store at which step collision occurred
        ray_trajectories = [ray_origins.clone()]  # Store as list, will concatenate later
        
        # Store previous positions for segment checking
        ray_prev_positions = ray_origins.clone()
        
        print(f"Starting ray march with {num_vertices} rays...")
        print(f"  Step size: {step_size}")
        print(f"  Algorithm: Ray-segment intersection (checks full paths)")
        
        for step in range(max_steps):
            if not ray_active.any():
                print(f"All rays stopped at step {step}")
                break
            
            # March active rays
            ray_prev_positions = ray_positions.clone()
            ray_positions[ray_active] += step_size * ray_directions[ray_active]
            
            # Store trajectory
            ray_trajectories.append(ray_positions.clone())
            
            # Check for ray-segment intersections
            active_indices = ray_active.nonzero().squeeze(1)
            num_active = active_indices.shape[0]
            
            if num_active < 2:
                break
            
            # For each active ray, check if it intersects with segments of OTHER active rays
            new_collisions = []
            
            for i in range(num_active):
                ray_i = active_indices[i].item()
                ray_pos = ray_positions[ray_i]
                ray_dir = ray_directions[ray_i]
                ray_origin = ray_origins[ray_i]
                
                for j in range(i + 1, num_active):
                    ray_j = active_indices[j].item()
                    
                    # Check if ray_i intersects with the last segment of ray_j
                    seg_start = ray_prev_positions[ray_j]
                    seg_end = ray_positions[ray_j]
                    
                    intersection, valid1, t_ray, t_seg = self.ray_intersects_segment(
                        ray_origin, ray_dir, seg_start, seg_end
                    )
                    
                    # Also check if ray_j intersects with the last segment of ray_i
                    seg_start2 = ray_prev_positions[ray_i]
                    seg_end2 = ray_positions[ray_i]
                    ray_origin2 = ray_origins[ray_j]
                    ray_dir2 = ray_directions[ray_j]
                    
                    intersection2, valid2, t_ray2, t_seg2 = self.ray_intersects_segment(
                        ray_origin2, ray_dir2, seg_start2, seg_end2
                    )
                    
                    if valid1 or valid2:
                        # Use the closer intersection
                        if valid1 and valid2:
                            int_point = intersection if t_ray < t_ray2 else intersection2
                        elif valid1:
                            int_point = intersection
                        else:
                            int_point = intersection2
                        
                        new_collisions.append((ray_i, ray_j, int_point))
            
            # Process collisions
            if len(new_collisions) > 0:
                for ray_i, ray_j, int_point in new_collisions:
                    collision_points.append(int_point)
                    collision_pairs.append(torch.tensor([ray_i, ray_j], device=self.device))
                    collision_steps.append(step)
                    
                    # Deactivate collided rays
                    ray_active[ray_i] = False
                    ray_active[ray_j] = False
                
                print(f"  Step {step}: {len(new_collisions)} new collision(s), "
                      f"{ray_active.sum().item()} rays remaining")
            elif step % 100 == 0:
                print(f"  Step {step}: {ray_active.sum().item()} rays active")
        
        # Consolidate results
        if len(collision_points) > 0:
            all_intersections = torch.stack(collision_points, dim=0)
            all_pairs = torch.stack(collision_pairs, dim=0)
        else:
            all_intersections = torch.zeros(0, 2, device=self.device)
            all_pairs = torch.zeros(0, 2, dtype=torch.long, device=self.device)
        
        # Concatenate trajectories (num_vertices, num_steps, 2)
        ray_trajectories = torch.stack(ray_trajectories, dim=1)
        
        print(f"\nRay marching complete!")
        print(f"  Found {all_intersections.shape[0]} Voronoi vertices")
        print(f"  Final active rays: {ray_active.sum().item()}")
        
        self.voronoi_vertices = all_intersections
        self.collision_data = {
            'voronoi_vertices': all_intersections,
            'collision_pairs': all_pairs,
            'collision_steps': collision_steps,
            'ray_trajectories': ray_trajectories,
            'ray_origins': ray_origins,
            'ray_directions': ray_directions,
            'ray_final_active': ray_active
        }
        
        return self.collision_data
    
    def build_voronoi_graph(self, intersection_data=None):
        """
        Build graph structure from segment intersections.
        
        Args:
            intersection_data: Output from find_segment_intersections() (optional if already stored)
        
        Returns:
            dict with:
                - vertices: All vertices (original + Voronoi)
                - edges: Voronoi edges connecting vertices
        """
        if intersection_data is None:
            if not hasattr(self, 'intersection_data') or self.intersection_data is None:
                raise ValueError("Must call find_segment_intersections() first or pass intersection_data")
            intersection_data = self.intersection_data
        else:
            self.intersection_data = intersection_data
        
        voronoi_verts = intersection_data['voronoi_vertices']
        voronoi_pairs = intersection_data['voronoi_pairs']
        
        if len(voronoi_verts) == 0:
            print("Warning: No Voronoi vertices found")
            return {
                'vertices': self.vertices,
                'edges': [],
                'num_original': self.vertices.shape[0],
                'num_voronoi': 0
            }
        
        # Combine original vertices and Voronoi vertices
        all_vertices = torch.cat([self.vertices, torch.stack(voronoi_verts)], dim=0)
        
        # Build edges: each Voronoi vertex connects the two original vertices
        # that created it
        edges = []
        
        for i, (v1, v2) in enumerate(voronoi_pairs):
            voronoi_v_idx = self.vertices.shape[0] + i
            
            # Edge from original vertex v1 to Voronoi vertex
            edges.append((v1, voronoi_v_idx))
            # Edge from original vertex v2 to Voronoi vertex
            edges.append((v2, voronoi_v_idx))
        
        self.voronoi_edges = edges
        
        return {
            'vertices': all_vertices,
            'edges': edges,
            'num_original': self.vertices.shape[0],
            'num_voronoi': len(voronoi_verts)
        }
    
    def get_edge_voronoi_cells(self):
        """
        Determine which Voronoi vertices belong to which original edge.
        
        Returns:
            dict mapping edge_idx -> list of Voronoi vertex indices
        """
        if self.collision_data is None:
            raise ValueError("Must run march_rays() first")
        
        collision_pairs = self.collision_data['collision_pairs']
        num_edges = self.edges.shape[0]
        
        edge_to_voronoi = {i: [] for i in range(num_edges)}
        
        # For each Voronoi vertex (intersection), determine which edge it represents
        for i, (v1, v2) in enumerate(collision_pairs):
            v1, v2 = v1.item(), v2.item()
            
            # Find edges incident to both v1 and v2
            edges_v1 = set(self.vertex_to_edges[v1])
            edges_v2 = set(self.vertex_to_edges[v2])
            
            # The shared edge is the one we're bisecting
            shared_edges = edges_v1 & edges_v2
            
            if len(shared_edges) == 1:
                edge_idx = list(shared_edges)[0]
                edge_to_voronoi[edge_idx].append(i)
            else:
                # Adjacent edges - this Voronoi vertex is on the boundary between them
                for edge_idx in edges_v1 | edges_v2:
                    edge_to_voronoi[edge_idx].append(i)
        
        return edge_to_voronoi


def polygons_to_tensor_list(polygons):
    """
    Convert various polygon formats to list of torch tensors.
    
    Args:
        polygons: numpy arrays, torch tensors, or list of either
    
    Returns:
        List of torch.Tensors
    """
    result = []
    
    if isinstance(polygons, (list, tuple)):
        for poly in polygons:
            if isinstance(poly, np.ndarray):
                result.append(torch.from_numpy(poly).float())
            elif isinstance(poly, torch.Tensor):
                result.append(poly.float())
            else:
                raise ValueError(f"Unknown polygon type: {type(poly)}")
    else:
        raise ValueError("polygons must be a list or tuple")
    
    return result

