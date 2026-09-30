use crate::mesh::{Edge, Polygon, Vertex};
use crate::mesh;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

struct TempVertex<'a> {
    vertex: &'a mut Vertex,
    connected_vertices: Vec<Weak<RefCell<TempVertex<'a>>>>,
    is_suspended: bool,
    current_y: f32,
    current_x: f32,
    last_deviation: f32
}

impl<'a> TempVertex <'a> {
    fn new(vertex: &'a mut Vertex, is_suspended: bool) -> Self {
        let current_y = vertex.y as f32;
        let current_x = vertex.x as f32;
        TempVertex { 
            vertex, 
            connected_vertices: Vec::new(), 
            is_suspended,
            current_y,
            current_x,
            last_deviation: 0.0
        }
    }
    // connected_vertices: impl Iterator<Item = &TempVertex<'_>>
    fn smooth(&mut self) {
        if !self.is_suspended {
            return;
        }
        let mut y_sum = 0.0;
        let mut x_sum = 0.0;
        for connected_vertex_ref in &self.connected_vertices {
            let connected_vertex = connected_vertex_ref;
            y_sum += connected_vertex.upgrade().expect("Original reference lost").borrow().current_y;
            x_sum += connected_vertex.upgrade().expect("Original reference lost").borrow().current_x;
        }
        let connected_len = self.connected_vertices.len() as f32;
        let new_y = y_sum / connected_len;
        let new_x = x_sum / connected_len;
        let new_deviation = (self.current_y - new_y).powi(2) + (self.current_x - new_x).powi(2);
        self.last_deviation = new_deviation;
        self.current_y = new_y;
        self.current_x = new_x;
    }
    fn bind(&mut self) {
        self.vertex.y = self.current_y as usize;
        self.vertex.x = self.current_x as usize;
    }
}

fn get_managed_vertices <'a>(vertices: &'a mut Vec<Vertex>, edges: &Vec<Edge>) -> Vec<Rc<RefCell<TempVertex<'a>>>>{ 
    let mut managed_vertices = Vec::with_capacity(vertices.len());
    for vertex in vertices {
        let mut vertex_edges = vertex.edges.iter().map(|ei| &edges[*ei]);
        let vertex_is_suspended = vertex_edges.all(|e| e.polygons.len() > 1);
        managed_vertices.push(Rc::new(
            RefCell::new(TempVertex::new(vertex, vertex_is_suspended))
        ));
    }
    let mut manager_refs: Vec<Weak<RefCell<TempVertex>>> = Vec::with_capacity(managed_vertices.len());
    {
        for manager in &managed_vertices {
            manager_refs.push(Rc::downgrade(manager));
        }
    }
    for managed_vertex_ref in managed_vertices.iter_mut().map(|r| Rc::clone(&r)) {
        let connected_vertices: Vec<Weak<RefCell<TempVertex>>>;
        {
            let vertex_fetch_ref = Rc::clone(&managed_vertex_ref);
            let vertex_fetch_borrow = vertex_fetch_ref.borrow();
            let vertex_edges = vertex_fetch_borrow.vertex.edges.iter()
                .map(|ei| &edges[*ei]);
            let connected_and_self_ids: Vec<usize> = vertex_edges.map(|e| vec![e.vertex_1, e.vertex_2]).
            flatten().collect();
            let connected_ids = connected_and_self_ids.iter().filter(|vi| **vi != managed_vertex_ref.borrow().vertex.id);
            connected_vertices = connected_ids.map(|vi| Weak::clone(&manager_refs[*vi])).collect();
        }
        let mut managed_vertex = managed_vertex_ref;
        managed_vertex.borrow_mut().connected_vertices.extend(connected_vertices);
    }
    managed_vertices
}

fn smooth_list(managed_vertices: &mut Vec<Rc<RefCell<TempVertex>>>) {
    for managed_vertex_ref in &mut *managed_vertices {
        managed_vertex_ref.borrow_mut().smooth();
    }
}

fn get_max_displacement(managers: &Vec<Rc<RefCell<TempVertex>>>) -> f32 {
    let mut max_displacement = 0.0;
    for manager in managers {
        let manager_displacement = manager.borrow().last_deviation;
        if max_displacement < manager_displacement {
            max_displacement = manager_displacement;
        }
    }
    max_displacement
}


pub fn smooth_polygons(
    polygon_coords: Vec<Vec<(usize, usize)>>,
    convergence_ratio: f32
) -> (Vec<Polygon>, Vec<Vertex>) {
    // Given a nested list of coordinate pairs, representing either triangles or quadrilaterals on a 2D plane, and combined into a mesh with shared edges, make cuts to turn all triangles into quadrilaterals
    assert!(polygon_coords.iter().all(|pv| pv.len() == 4 || pv.len() == 3));
    let mut vertex_list = mesh::get_vertices(&polygon_coords);
    let mut edge_list = mesh::get_edge_vector(&polygon_coords, &vertex_list);
    let mut polygons = mesh::get_polygons(&polygon_coords, &vertex_list, &edge_list);
    mesh::connect_mesh(&mut vertex_list, &mut edge_list, &polygons);
    let mut managed_vertices = get_managed_vertices(&mut vertex_list, &edge_list);
    smooth_list(&mut managed_vertices);
    let average_first_displacement = managed_vertices.iter().filter(|m| m.borrow().is_suspended)
    .map(|m| m.borrow().last_deviation).sum::<f32>() / (managed_vertices.len() as f32);
    let target_max_displacement = average_first_displacement * convergence_ratio;
    let mut max_current_displacement = get_max_displacement(&managed_vertices);
    while max_current_displacement > target_max_displacement {
        smooth_list(&mut managed_vertices);
        max_current_displacement = get_max_displacement(&managed_vertices);
    }
    for manager in managed_vertices {
        manager.borrow_mut().bind()
    }
    (polygons, vertex_list)
}
