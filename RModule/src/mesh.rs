use std::fmt::Display;
use std::cmp::Ordering;
use crate::double_key_tree::DoubleKeyTree;
use crate::find_index_in;
use crate::find_index_in::BinaryKeyOrd;

pub struct Vertex {
    pub id: usize,
    pub y: usize,
    pub x: usize
}

impl BinaryKeyOrd for Vertex {
    fn search_on(&self, key_1: usize, key_2: usize) -> Ordering {
        match self.y.cmp(&key_1) {
            Ordering::Equal => {}
            ord => return ord
        }
        self.x.cmp(&key_2)
    }
}

impl Display for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.y, self.x)
    }
}

pub struct Edge {
    pub id: usize,
    pub vertex_1: usize,
    pub vertex_2: usize,
    pub polygons: Vec<usize>
}

impl Edge {
    fn add_connected_polygon(&mut self, polygon_id: usize) {
        if ! self.polygons.iter().any(|p| *p == polygon_id)
        {
            let current_len = self.polygons.len();
            assert!(current_len < 2);
            self.polygons.push(polygon_id);
        }
    }
}

impl BinaryKeyOrd for Edge {
    fn search_on(&self, key_1: usize, key_2: usize) -> Ordering {
        match self.vertex_1.cmp(&key_1) {
            Ordering::Equal => {}
            ord => return ord
        }
        self.vertex_2.cmp(&key_2)
    }
}

impl Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.vertex_1, self.vertex_2)
    }
}

pub struct Polygon {
    pub id: usize,
    pub original_id: usize,
    pub vertices: Vec<usize>,
    pub edges: Vec<usize>
}

pub fn get_vertices(polygons: &Vec<Vec<(usize, usize)>>) -> Vec<Vertex> {
    // Parses a nested list of polygon vertex coordinates into vertex data objects
    let mut tree = DoubleKeyTree::new();
    for polygon_vector in polygons.iter() {
        for polygon_vertex in polygon_vector.iter() {
            tree.insert_to_tree(polygon_vertex.0, polygon_vertex.1);
        }
    }
    let vertex_vector = tree.get_order();
    let vertex_count = vertex_vector.len() / 2;
    let mut vertices = Vec::with_capacity(vertex_count);
    for vertex_index in 0..vertex_count {
        vertices.push(Vertex{
            id: vertex_index, 
            y: vertex_vector[vertex_index * 2], 
            x: vertex_vector[vertex_index * 2 + 1]
        });
    }
    vertices
}

impl Polygon {
    pub fn is_triangle(&self) -> bool {
        self.vertices.len() == 3
    }
    pub fn is_edge_void_border(&self, polygon_edge_index: usize, edges: &Vec<Edge>) -> bool {
        // Indicates whether the provided edge is contained only by this polygon
        let edge = &edges[self.edges[polygon_edge_index]];
        if edge.polygons.len() == 1 {
            if edge.polygons[0] == self.id {
                return true;
            }
            else {
                panic!("Accessed unowned edge in is_void_border check")
            }
        }
        else {
            false
        }
    }
    pub fn is_void_border(&self, edges: &Vec<Edge>) -> bool {
        // Indicates whether this polygon has one or more edges that make up only this polygon
        self.get_connections(edges).iter().any(|c| c.is_none())
    }
    pub fn is_target(&self, edges: &Vec<Edge>) -> bool {
        // Indicates whether this polygon can be used as the terminal point of a cut
        self.is_triangle() || self.is_void_border(edges)
    }
    pub fn get_connections(&self, edges: &Vec<Edge>) -> Vec<Option<usize>> {
        // Provides a list of polygon ids bordering this polygon.  Polygon ids are listed in the order of their connecting edge in this polygon's edge list, and are None if the edge is a void border
        let mut results = Vec::with_capacity(self.edges.len());
        for edge_id in &self.edges {
            let edge = &edges[*edge_id];
            let non_self_edge_connections: Vec<&usize> = edge.polygons.iter().filter(|connected_id| **connected_id != self.id).collect();
            if non_self_edge_connections.len() == 0 {
                results.push(None);
            }
            else if non_self_edge_connections.len() == 1 {
                results.push(Some(*non_self_edge_connections[0]));
            }
            else {
                panic!("Invalid mesh geometry found when getting polygon connections");
            }
        }
        results
    }
    
}


pub fn get_edge_vertex_indices(polygon_vect: &Vec<(usize, usize)>, point_count: usize, point_index: usize, vertex_vector: &Vec<Vertex>) -> (usize, usize) {
    // Scans a list of polygon vertex coordinates to identify the first and second vertex index for the edge at a provided location in the polygon
    let point_y_1 = polygon_vect[point_index].0;
    let point_x_1 = polygon_vect[point_index].1;
    let vertex_index_1 = find_index_in::find_index_in(point_y_1, point_x_1, &vertex_vector).expect("Point one missing in vector");
    let point_y_2 = polygon_vect[(point_index + 1) % point_count].0;
    let point_x_2 = polygon_vect[(point_index + 1) % point_count].1;
    let vertex_index_2 = find_index_in::find_index_in(point_y_2, point_x_2, &vertex_vector).expect("Point two missing in vector");
    if vertex_index_1 > vertex_index_2 {
        (vertex_index_1, vertex_index_2)
    } else {
        (vertex_index_2, vertex_index_1)
    }
}

pub fn get_edge_vector(polygons: &Vec<Vec<(usize, usize)>>, vertex_vector: &Vec<Vertex>) -> Vec<Edge> {
    // Parses a nested list of polygon vertex coordinates into edge data objects
    let mut tree = DoubleKeyTree::new();
    for polygon_vect in polygons.iter() {
        let point_count = polygon_vect.len();
        for point_index in 0..point_count {
            let (vertex_1, vertex_2) = get_edge_vertex_indices(polygon_vect, point_count, point_index, vertex_vector);
            tree.insert_to_tree(vertex_1, vertex_2);
        }
    }
    let edge_vector = tree.get_order();
    let edge_count = edge_vector.len() / 2;
    let mut edges = Vec::with_capacity(edge_count);
    for edge_index in 0..edge_count {
        edges.push(Edge {
            id: edge_index, 
            vertex_1: edge_vector[edge_index * 2], 
            vertex_2: edge_vector[edge_index * 2 + 1],
            polygons: Vec::new()
        });
    }
    edges
}


pub fn get_polygons(polygons_coords: &Vec<Vec<(usize, usize)>>, total_vertices: &Vec<Vertex>, total_edges: &Vec<Edge>) -> Vec<Polygon> {
    // Parses a nested list of polygon vertex coordinates into polygon data objects
    let mut polygons: Vec<Polygon> = Vec::with_capacity(polygons_coords.len());
    for (polygon_id, polygon_vect) in polygons_coords.iter().enumerate() {
        let point_count = polygon_vect.len();
        let mut vertex_vector = Vec::with_capacity(point_count);
        let mut edge_vector = Vec::with_capacity(point_count);
        for (point_index, point_coords) in polygon_vect.iter().enumerate() {
            let vertex_index = find_index_in::find_index_in(point_coords.0, point_coords.1, total_vertices).expect("Polygon vertex not found");
            let (edge_vertex_1, edge_vertex_2) = get_edge_vertex_indices(polygon_vect, point_count, point_index, total_vertices);
            let edge_index = find_index_in::find_index_in(edge_vertex_1, edge_vertex_2, total_edges).expect("Edge not found");
            vertex_vector.push(vertex_index);
            edge_vector.push(edge_index);
        }
            polygons.push(Polygon { id: polygon_id, original_id: polygon_id, vertices: vertex_vector, edges: edge_vector });
    }
    polygons
}

pub fn connect_mesh(total_edges: &mut Vec<Edge>, polygons: &Vec<Polygon>) {
    // Connects the edge objects to the polygon objects
    for polygon in polygons {
        for polygon_edge in &polygon.edges {
            total_edges[*polygon_edge].add_connected_polygon(polygon.id);
        }
    }
}