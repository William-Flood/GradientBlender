
use crate::mesh::{Edge, Polygon, Vertex};
use crate::mesh;

#[derive(Clone, Copy)]
struct CutPlanPreviousStepReference {
    edge_cut_to: usize,
    edge_cut_from: usize,
    last_step_index: usize
}

#[derive(Clone, Copy)]
struct CutPlanStep<'a> {
    butcher: PolygonButcher<'a>,
    previous_step_reference: Option<CutPlanPreviousStepReference>
}

struct TriangleCutPlanner<'a> {
    cut_paths: Vec<Vec<CutPlanStep <'a>>>,
    search_mask: Vec<bool>
}

pub struct TempPolygon {
    pub ys: Vec<usize>,
    pub xs: Vec<usize>,
    pub cut_from: usize,
    pub original_rect: usize
}

#[derive(Clone, Copy)]
struct PolygonButcher<'a> {
    polygon: &'a Polygon
}

impl<'a> PolygonButcher<'a> {
    fn get_connections(&self, edges: &Vec<Edge>) -> Vec<Option<usize>> {
        self.polygon.get_connections(edges)
    }
    pub fn end_cut(&self, cut_edge: usize, vertices: &Vec<Vertex>, edges: &Vec<Edge>) -> Vec<TempPolygon> {
        // Produces a data object to regenerate the mesh after cutting from this polygon
        if self.polygon.vertices.len() == 3 {
            let mut cut_result = TempPolygon {
                ys: Vec::with_capacity(self.polygon.vertices.len()),
                xs: Vec::with_capacity(self.polygon.vertices.len()),
                cut_from: self.polygon.id,
                original_rect: self.polygon.original_id
            };
            for (polygon_edge_index, vertex_index) in self.polygon.vertices.iter().enumerate() {
                let vertex = &vertices[*vertex_index];
                cut_result.ys.push(vertex.y);
                cut_result.xs.push(vertex.x);
                if cut_edge == polygon_edge_index {
                    let next_vertex_index = self.polygon.vertices[(polygon_edge_index + 1) % 3];
                    let next_vertex = &vertices[next_vertex_index];
                    cut_result.ys.push((vertex.y + next_vertex.y) / 2);
                    cut_result.xs.push((vertex.x + next_vertex.x) / 2);
                }
            }
            return vec![cut_result]
        }
        else {
            let second_cut;
            if self.polygon.is_edge_void_border((cut_edge + 2) % 4, edges) {
                second_cut = (cut_edge + 2) % 4;
            }
            else {
                let second_cut_offset_options = vec![1, 3];
                let second_cut_find = second_cut_offset_options.iter().find(|o| self.polygon.is_edge_void_border((cut_edge + *o) % 4, edges));
                if let Some(second_cut_offset) = second_cut_find {
                    second_cut = (cut_edge + second_cut_offset) % 4;
                }
                else {
                    panic!("Invalid end polygon chosen - not triangle or void border");
                }
            }
            return self.cut(cut_edge, second_cut, vertices);
        }
    }
    pub fn cut(&self, cut_edge_1: usize, cut_edge_2: usize, vertices: &Vec<Vertex>) -> Vec<TempPolygon> {
        // Produces a data object to regenerate the mesh after cutting through this polygon
        if self.polygon.vertices.len() != 4 {
            panic!("Triangle cut called on non-quadrilateral")
        }
        let vertex_objs: Vec<&Vertex> = self.polygon.vertices.iter().map(|vi| &vertices[*vi]).collect();
        let cut_vertex_1_y = (vertex_objs[cut_edge_1].y + vertex_objs[(cut_edge_1 + 1) % 4].y) / 2;
        let cut_vertex_1_x = (vertex_objs[cut_edge_1].x + vertex_objs[(cut_edge_1 + 1) % 4].x) / 2;
        let cut_vertex_2_y = (vertex_objs[cut_edge_2].y + vertex_objs[(cut_edge_2 + 1) % 4].y) / 2;
        let cut_vertex_2_x = (vertex_objs[cut_edge_2].x + vertex_objs[(cut_edge_2 + 1) % 4].x) / 2;
        if (cut_edge_2 + 4 - cut_edge_1) % 4 == 2 {
            let cut_poly_1_ys = vec![cut_vertex_1_y, vertex_objs[(cut_edge_1 + 1) % 4].y, vertex_objs[(cut_edge_1 + 2) % 4].y, cut_vertex_2_y];
            let cut_poly_1_xs = vec![cut_vertex_1_x, vertex_objs[(cut_edge_1 + 1) % 4].x, vertex_objs[(cut_edge_1 + 2) % 4].x, cut_vertex_2_x];
            let cut_result_1 = TempPolygon {
                ys: cut_poly_1_ys,
                xs: cut_poly_1_xs,
                cut_from: self.polygon.id,
                original_rect: self.polygon.original_id
            };
            let cut_poly_2_ys = vec![cut_vertex_2_y, vertex_objs[(cut_edge_1 + 3) % 4].y, vertex_objs[cut_edge_1].y, cut_vertex_1_y];
            let cut_poly_2_xs = vec![cut_vertex_2_x, vertex_objs[(cut_edge_1 + 3) % 4].x, vertex_objs[cut_edge_1].x, cut_vertex_1_x];
            let  cut_result_2 = TempPolygon {
                ys: cut_poly_2_ys,
                xs: cut_poly_2_xs,
                cut_from: self.polygon.id,
                original_rect: self.polygon.original_id
            };
            return vec![cut_result_1, cut_result_2];
        }
        else {
            let cut_vertex_3_y = vertex_objs.iter().map(|v| v.y).sum::<usize>() / 4;
            let cut_vertex_3_x = vertex_objs.iter().map(|v| v.x).sum::<usize>() / 4;
            if (cut_edge_2 + 4 - cut_edge_1) % 4 == 1 {
                let cut_poly_1_ys = vec![vertex_objs[cut_edge_1].y, cut_vertex_1_y, cut_vertex_3_y, vertex_objs[(cut_edge_1 + 3) % 4].y];
                let cut_poly_1_xs = vec![vertex_objs[cut_edge_1].x, cut_vertex_1_x, cut_vertex_3_x, vertex_objs[(cut_edge_1 + 3) % 4].x];
                let cut_result_1 = TempPolygon {
                    ys: cut_poly_1_ys,
                    xs: cut_poly_1_xs,
                    cut_from: self.polygon.id,
                    original_rect: self.polygon.original_id
                };
                let cut_poly_2_ys = vec![cut_vertex_1_y,  vertex_objs[(cut_edge_1 + 1) % 4].y, cut_vertex_2_y, cut_vertex_3_y];
                let cut_poly_2_xs = vec![cut_vertex_1_x,  vertex_objs[(cut_edge_1 + 1) % 4].x, cut_vertex_2_x, cut_vertex_3_x];
                let cut_result_2 = TempPolygon {
                    ys: cut_poly_2_ys,
                    xs: cut_poly_2_xs,
                    cut_from: self.polygon.id,
                    original_rect: self.polygon.original_id
                };
                let cut_poly_3_ys = vec![cut_vertex_2_y,  vertex_objs[(cut_edge_1 + 2) % 4].y, vertex_objs[(cut_edge_1 + 3) % 4].y, cut_vertex_3_y];
                let cut_poly_3_xs = vec![cut_vertex_2_x,  vertex_objs[(cut_edge_1 + 2) % 4].x, vertex_objs[(cut_edge_1 + 3) % 4].x, cut_vertex_3_x];
                let cut_result_3 = TempPolygon {
                    ys: cut_poly_3_ys,
                    xs: cut_poly_3_xs,
                    cut_from: self.polygon.id,
                    original_rect: self.polygon.original_id

                };
                return vec![cut_result_1, cut_result_2, cut_result_3];
            }
            else {
                let cut_poly_1_ys = vec![vertex_objs[cut_edge_1].y, cut_vertex_1_y, cut_vertex_3_y, cut_vertex_2_y];
                let cut_poly_1_xs = vec![vertex_objs[cut_edge_1].x, cut_vertex_1_x, cut_vertex_3_x, cut_vertex_2_x];
                let cut_result_1 = TempPolygon {
                    ys: cut_poly_1_ys,
                    xs: cut_poly_1_xs,
                    cut_from: self.polygon.id,
                    original_rect: self.polygon.original_id
                };
                let cut_poly_2_ys = vec![cut_vertex_1_y,  vertex_objs[(cut_edge_1 + 1) % 4].y, vertex_objs[(cut_edge_1 + 2) % 4].y, cut_vertex_3_y];
                let cut_poly_2_xs = vec![cut_vertex_1_x,  vertex_objs[(cut_edge_1 + 1) % 4].x, vertex_objs[(cut_edge_1 + 2) % 4].x, cut_vertex_3_x];
                let cut_result_2 = TempPolygon {
                    ys: cut_poly_2_ys,
                    xs: cut_poly_2_xs,
                    cut_from: self.polygon.id,
                    original_rect: self.polygon.original_id
                };
                let cut_poly_3_ys = vec![vertex_objs[(cut_edge_1 + 2) % 4].y, vertex_objs[(cut_edge_1 + 3) % 4].y, cut_vertex_2_y, cut_vertex_3_y];
                let cut_poly_3_xs = vec![vertex_objs[(cut_edge_1 + 2) % 4].x, vertex_objs[(cut_edge_1 + 3) % 4].x, cut_vertex_2_x, cut_vertex_3_x];
                let cut_result_3 = TempPolygon {
                    ys: cut_poly_3_ys,
                    xs: cut_poly_3_xs,
                    cut_from: self.polygon.id,
                    original_rect: self.polygon.original_id
                };
                return vec![cut_result_1, cut_result_2, cut_result_3];
            }
        }
    }
    pub fn to_quad_if_void_border(&self, vertices: &Vec<Vertex>, edges: &Vec<Edge>) -> Option<TempPolygon> {
        if self.polygon.vertices.len() == 3 {
            if let Some(void_border) = self.polygon.get_connections(edges).iter().position(|c| c.is_none()) {
                let cut_y = (vertices[self.polygon.vertices[void_border]].y + vertices[self.polygon.vertices[(void_border + 1) % 3]].y) /2;
                let cut_x = (vertices[self.polygon.vertices[void_border]].x + vertices[self.polygon.vertices[(void_border + 1) % 3]].x) /2;
                let new_ys = vec![
                    vertices[self.polygon.vertices[void_border]].y, 
                    cut_y, 
                    vertices[self.polygon.vertices[(void_border + 1) % 3]].y, 
                    vertices[self.polygon.vertices[(void_border + 2) % 3]].y
                ];
                let new_xs = vec![vertices[
                    self.polygon.vertices[void_border]].x, 
                    cut_x, 
                    vertices[self.polygon.vertices[(void_border + 1) % 3]].x, 
                    vertices[self.polygon.vertices[(void_border + 2) % 3]].x
                ];
                return Some(TempPolygon { ys: new_ys, xs: new_xs, cut_from: self.polygon.id, original_rect: self.polygon.original_id });
            }
            else {
                return None;
            }
        }
        else {
            return None;
        }
    }
}

impl<'a> TriangleCutPlanner<'a> {
    fn advance_cut_plan(&mut self, polygons: &'a Vec<Polygon>, edges: &Vec<Edge>) {
        // Expands the search for a cut by one layer
        let mut next_search_layer: Vec<CutPlanStep> = Vec::new();
        let search_boundary = self.cut_paths.last().expect("get_search_boundary called on an uninitialized TriangleCutPlanner");
        for (search_node_index, search_node) in search_boundary.iter().enumerate() {
            for (search_polygon_edge_index, connection) in search_node.butcher.get_connections(edges).iter().enumerate() {
                let connected_polygon_index = connection.expect("Navigated from void boundary");
                if !self.search_mask[connected_polygon_index] {
                    self.search_mask[connected_polygon_index] = true;
                    let polygon_to = &polygons[connected_polygon_index];
                    next_search_layer.push(CutPlanStep { 
                        butcher: PolygonButcher { polygon: polygon_to }, 
                        previous_step_reference: Some(CutPlanPreviousStepReference{
                            last_step_index: search_node_index,
                            edge_cut_from: search_polygon_edge_index,
                            edge_cut_to: polygon_to.edges.iter().position(|e| *e == search_node.butcher.polygon.edges[search_polygon_edge_index])
                                .expect("Attempting to find the edge on a polygon that doesn't contain it")
                        }) 
                    });
                }

            }
        }
        self.cut_paths.push(next_search_layer);
    }
    fn new(start_polygon: &'a Polygon, polygon_count: usize) -> Self {
        let mut search_mask = vec![false; polygon_count];
        search_mask[start_polygon.id] = true;
        TriangleCutPlanner { cut_paths:  vec![vec![CutPlanStep{butcher: PolygonButcher { polygon: start_polygon }, previous_step_reference: None}]], search_mask }
    }
    fn get_plan(&self, edges: &Vec<Edge>) -> Option<Vec<CutPlanStep<'a >>> {
        // Navigates from a target polygon in the outermost search layer to the start polygon.
        let mut plan = Vec::with_capacity(self.cut_paths.len());
        // Note: Cuts starts from the last set of polygons found by the planner and end with the starting polygon of the planner
        let start_cut_res = self.cut_paths.last()
            .expect("get_plan called on an uninitialized TriangleCutPlanner")
            .iter().find(|p| p.butcher.polygon.is_target(edges));
        if let Some(start_cut) = start_cut_res {
            plan.push(*start_cut);
            let mut current_cut = start_cut;
            let mut current_plan_layer_index = self.cut_paths.len() - 1;
            while current_plan_layer_index > 0 {
                current_plan_layer_index -= 1;
                let current_plan_layer = &self.cut_paths[current_plan_layer_index];
                let current_cut_index = current_cut.previous_step_reference.expect("Intermediate path node in cut did not reference its predecessor").last_step_index;
                current_cut = &current_plan_layer[current_cut_index];
                plan.push(*current_cut);
            }
            return Some(plan);
        }
        else {
            return None;
        }
    }
}

fn find_cut<'a>(polygons: &'a Vec<Polygon>, edges: &Vec<Edge>) -> Vec<CutPlanStep<'a >> {
    // Examines a mesh to find the closest pair of either two triangles or one triangle or a void border, and identifies the sequence across the mesh to cut from one to the other
    let mut planners: Vec<TriangleCutPlanner> = polygons.iter().filter(|p| p.is_triangle())
    .map(|p| TriangleCutPlanner::new(p, polygons.len()))
    .collect();
    let mut search_countdown= polygons.len() as i32;
    while search_countdown >= 0 {
        for planner in planners.iter_mut() {
            planner.advance_cut_plan(polygons, edges);
            if let Some(cut_result) = planner.get_plan(edges) {
                return cut_result;
            }
        }
        search_countdown -= 1;
    }
    panic!("Exhausted search for cut");
}


fn get_cut_trianges(cut_steps: Vec<CutPlanStep>, vertices: &Vec<Vertex>, edges: &Vec<Edge>) -> Vec<TempPolygon> {
    // Given a plan to cut the mesh, generate the new polygons 
    let mut cut_from = None;
    let mut cut_results = Vec::new();
    for step in cut_steps {
        if let Some(cut_from_val) = cut_from {
            if let Some(connection_to_last) = step.previous_step_reference {
                // Middle cut
                cut_results.append(&mut step.butcher.cut(cut_from_val, connection_to_last.edge_cut_to, vertices));
                cut_from = Some(connection_to_last.edge_cut_from);
            }
            else {
                // Final cut
                cut_results.append(&mut step.butcher.end_cut(cut_from_val, vertices, edges));
            }
        }
        else {
            if let Some(connection_to_last) = step.previous_step_reference {
                // First cut
                cut_results.append(&mut step.butcher.end_cut(connection_to_last.edge_cut_to, vertices, edges));
                cut_from = Some(connection_to_last.edge_cut_from);
            }
            else {
                panic!("Invalid cut plan - no cuts to make");
            }
        }
    }
    cut_results
}



mod merge {
    use super::{TempPolygon, Vertex, Edge, Polygon};
    use crate::{find_index_in, mesh};

    struct VertexMerger<'a> {
        vertices: &'a mut Vec<Vertex>,
        old_ids: Vec<usize>,
        new_old_map: Option<Vec<usize>>
    }

    impl<'a> VertexMerger<'a> {
        fn new(vertices: &'a mut Vec<Vertex>) -> Self {
            let old_ids = vertices.iter().map(|v| v.id).collect();
            VertexMerger { vertices, old_ids, new_old_map: None }
        }
        fn insert (&mut self, y: usize, x: usize) {
            let new_id = self.vertices.len();
            find_index_in::insert_into_double_key_ordered(
                y,
                x,
                self.vertices,
                || Vertex { id: new_id, y: y, x: x, edges: Vec::new() }
            );
        }
        fn map_old_and_refresh_ids(&mut self) {
            let mut is_old = vec![false; self.vertices.len()];
            for old_id in &self.old_ids {
                is_old[*old_id] = true;
            }
            let mut new_old_map = Vec::with_capacity(self.old_ids.len());
            for (new_position, vertex) in self.vertices.iter_mut().enumerate() {
                if is_old[vertex.id] {
                    new_old_map.push(new_position);
                }
                vertex.id = new_position;
            }
            self.new_old_map = Some(new_old_map);
        }
        fn find_in(&self, y: usize, x:usize) -> Option<usize> {
            find_index_in::find_index_in(
                    y, 
                    x, 
                    self.vertices)
        }
    }

    
    struct EdgeMerger<'a> {
        edges: &'a mut Vec<Edge>,
        old_ids: Vec<usize>,
        new_old_map: Option<Vec<usize>>
    }

    impl<'a> EdgeMerger<'a> {
        fn new(edges: &'a mut Vec<Edge>) -> Self {
            let old_ids = edges.iter().map(|v| v.id).collect();
            EdgeMerger { edges, old_ids, new_old_map: None }
        }
        fn insert (&mut self, vertex_1: usize, vertex_2: usize) {
            let new_id = self.edges.len();
            find_index_in::insert_into_double_key_ordered(
                vertex_1,
                vertex_2,
                self.edges,
                || Edge { id: new_id, vertex_1, vertex_2, polygons: Vec::new() }
            );
        }
        fn map_old_and_refresh_ids(&mut self) {
            let mut is_old = vec![false; self.edges.len()];
            for old_id in &self.old_ids {
                is_old[*old_id] = true;
            }
            let mut new_old_map = Vec::with_capacity(self.old_ids.len());
            for (new_position, edge) in self.edges.iter_mut().enumerate() {
                if is_old[edge.id] {
                    new_old_map.push(new_position);
                }
                edge.id = new_position;
            }
            self.new_old_map = Some(new_old_map);
        }
        fn find_in(&self, vertex_1: usize, vertex_2: usize) -> Option<usize> {
            find_index_in::find_index_in(
                vertex_1,
                vertex_2,
                    self.edges)
        }
        fn remap_vertex_lists(&mut self, vertex_merger: &VertexMerger) {
            let new_old_index_map = vertex_merger.new_old_map.as_ref().expect("Vertex old to new map retrieved before binding");
            for edge in &mut *self.edges {
                edge.vertex_1 = new_old_index_map[edge.vertex_1];
                edge.vertex_2 = new_old_index_map[edge.vertex_2];
            }
        }
    }

    fn merge_vertices_from(vertex_merger: &mut VertexMerger, new_polygons: &Vec<TempPolygon>) {
        // Use the data generated for the polygons to add to the mesh to update the list of vertexes
        for new_polygon in new_polygons {
            for (y, x) in new_polygon.ys.iter().zip(new_polygon.xs.iter()) {
                vertex_merger.insert(*y, *x);
            }
        }
        vertex_merger.map_old_and_refresh_ids();
    }
    
    fn merge_edges_from(vertex_merger: &VertexMerger, edge_merger: &mut EdgeMerger, new_polygons: &Vec<TempPolygon>) {
        // Use the data generated for the polygons to add to the mesh to update the list of edges

        edge_merger.remap_vertex_lists(vertex_merger);
        for new_polygon in new_polygons {
            for edge_index in 0..(new_polygon.ys.len()) {
                let coord_array = new_polygon.ys.iter()
                .zip(new_polygon.xs.iter())
                .map(|pt| (*pt.0, *pt.1))
                .collect();
                let point_count = new_polygon.ys.len();
                let (vertex_1_id, vertex_2_id) = mesh::get_edge_vertex_indices(&coord_array, point_count, edge_index, vertex_merger.vertices);
                edge_merger.insert(vertex_1_id, vertex_2_id);
            }
        }
        edge_merger.map_old_and_refresh_ids();
    }
    
    fn merge_polygons_from(
            vertex_merger: &VertexMerger,
            edge_merger: &EdgeMerger, 
            polygons: &mut Vec<Polygon>,
            new_polygons: &Vec<TempPolygon>
        ) {
        // Use the data generated for the polygons to add to the mesh to update the list of  polygons
        let new_old_vertex_index_map = vertex_merger.new_old_map.as_ref().expect("Vertex old to new map retrieved before binding");
        let new_old_edge_index_map = edge_merger.new_old_map.as_ref().expect("Vertex old to new map retrieved before binding");
        for polygon in &mut *polygons {
            let new_polygon_vertices: Vec<usize> = polygon.vertices.iter().map(|vi| new_old_vertex_index_map[*vi]).collect();
            polygon.vertices = new_polygon_vertices;
            polygon.edges = polygon.edges.iter().map(|ei| new_old_edge_index_map[*ei]).collect();
        }

        let mut keep_mask = vec![true; polygons.len()];
        for new_polygon in new_polygons {
            keep_mask[new_polygon.cut_from] = false;
        }
        polygons.retain(|p| keep_mask[p.id]);

        for (new_id, polygon) in polygons.iter_mut().enumerate() {
            polygon.id = new_id;
        }

        for new_polygon in new_polygons {
            let coord_array: Vec<(usize, usize)> = new_polygon.ys.iter()
                .zip(new_polygon.xs.iter())
                .map(|pt| (*pt.0, *pt.1))
                .collect();
            let polygon_vertex_counts = new_polygon.ys.len();
            assert_eq!(new_polygon.ys.len(), new_polygon.xs.len());
            let mut polygon_vertexes = Vec::with_capacity(polygon_vertex_counts);
            let mut polygon_edges = Vec::with_capacity(polygon_vertex_counts);
            let new_id = polygons.len();
            for polygon_vertex_index in 0..(new_polygon.ys.len()) {
                let coord_y = coord_array[polygon_vertex_index].0;
                let coord_x = coord_array[polygon_vertex_index].1;
                let vertex_index = vertex_merger.find_in(coord_y, coord_x).expect("Vertex not added or could not be found");
                let point_count = new_polygon.ys.len();
                let (vertex_1_id, vertex_2_id) = mesh::get_edge_vertex_indices(&coord_array, point_count, polygon_vertex_index, vertex_merger.vertices);
                let edge_index = edge_merger.find_in(vertex_1_id, vertex_2_id).expect("Edge not added or could not be found");
                polygon_vertexes.push(vertex_index);
                polygon_edges.push(edge_index);
            }
            polygons.push(
                Polygon { id: new_id, original_id: new_polygon.original_rect, vertices: polygon_vertexes, edges: polygon_edges }
            );
        }

    }

    pub fn merge_new_polygons(new_polygons: Vec<TempPolygon>, vertices: &mut Vec<Vertex>, edges: &mut Vec<Edge>, polygons: &mut Vec<Polygon>) {
        // Use the data generated for the polygons to add to the mesh to update the list of vertexes, edges, and polygons
        let mut vertex_merger = VertexMerger::new(vertices);
        let mut edge_merger = EdgeMerger::new(edges);
        merge_vertices_from(&mut vertex_merger, &new_polygons);
        merge_edges_from(&vertex_merger, &mut edge_merger, &new_polygons);
        merge_polygons_from(&vertex_merger, &edge_merger, polygons, &new_polygons);
        for edge in &mut *edges {
            edge.polygons = Vec::new();
        }
        mesh::connect_mesh(vertices, edges, polygons);
    }
}

pub fn cut_out_trianges(
    polygon_coords: Vec<Vec<(usize, usize)>>
) -> (Vec<Polygon>, Vec<Vertex>) {
    // Given a nested list of coordinate pairs, representing either triangles or quadrilaterals on a 2D plane, and combined into a mesh with shared edges, make cuts to turn all triangles into quadrilaterals
    assert!(polygon_coords.iter().all(|pv| pv.len() == 4 || pv.len() == 3));
    let mut vertex_list = mesh::get_vertices(&polygon_coords);
    let mut edge_list = mesh::get_edge_vector(&polygon_coords, &vertex_list);
    let mut polygons = mesh::get_polygons(&polygon_coords, &vertex_list, &edge_list);
    mesh::connect_mesh(&mut vertex_list,  &mut edge_list, &polygons);
    let border_cuts: Vec<TempPolygon> = polygons.iter()
        .map(|p| PolygonButcher {polygon: p})
        .map(|pb| pb.to_quad_if_void_border(&vertex_list, &edge_list))
    .filter_map(|pt| pt).collect();
    merge::merge_new_polygons(border_cuts, &mut vertex_list, &mut edge_list, &mut polygons);
    assert!(!polygons.iter().any(|p| p.is_void_border(&edge_list) && p.is_triangle()));
    while polygons.iter().any(|p| p.is_triangle()) {
        let cut_steps = find_cut(&polygons, &edge_list);
        let new_polygons = get_cut_trianges(cut_steps, &vertex_list, &edge_list);
        assert_ne!(new_polygons.len(), 0);
        merge::merge_new_polygons(new_polygons, &mut vertex_list, &mut edge_list, &mut polygons);
    }

    (polygons, vertex_list)
}
