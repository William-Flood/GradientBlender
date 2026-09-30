use pyo3::types::PyList;
use crate::numpy_utils;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{Ix2, PyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use crate::mesh::{Polygon, Vertex};
use crate::double_key_tree;

pub fn get_polygon_vector(
    polygons: Bound<'_, PyList>
) -> PyResult<Vec<Vec<(usize, usize)>>> {
    let mut polygons_vector= Vec::with_capacity(polygons.len());
    for polygon_obj in polygons.iter() {
        let polygon_array = polygon_obj.cast::<PyArray<i32, Ix2>>()
            .expect("Unable to cast region point array").readonly();
        let point_count = *polygon_array.shape().get(1).ok_or("No point count!").
            map_err(|e| PyValueError::new_err(e.to_string()))?;
        let polygon_array_vect = numpy_utils::numpy_to_vec(&polygon_array);
        let mut polygon_vect = Vec::with_capacity(point_count);
        for point_index in 0..(point_count) {
            polygon_vect.push((polygon_array_vect[point_index] as usize, polygon_array_vect[point_index + point_count] as usize))
        }
        polygons_vector.push(polygon_vect);
    }
    Ok(polygons_vector)
}

#[pyclass]
pub struct PolygonResult {
    #[pyo3(get)]
    original_id: usize,
    #[pyo3(get)]
    points: Py<PyArray2<i32>>
}

pub fn get_polygon_results<'py>(
    py: Python<'py>,
    polygons: Vec<Polygon>,
    vertices: Vec<Vertex>
) -> PyResult<Bound<'py, PyList>> {
    let mut result_vector = Vec::with_capacity(polygons.len());
    for polygon in polygons.iter() {
        let vertex_count = polygon.vertices.len();
        let mut polygon_vertices = vec![0; vertex_count * 2];
        for (polygon_vertex_index, vertex_id) in polygon.vertices.iter().enumerate() {
            let vertex = &vertices[*vertex_id];
            polygon_vertices[polygon_vertex_index] = vertex.y as i32;
            polygon_vertices[polygon_vertex_index + vertex_count] = vertex.x as i32;
        }
        let point_array_flat = PyArray::from_vec(py, polygon_vertices);
        let point_array = point_array_flat.reshape((2, vertex_count)).expect("Reshape failed");
        let point_array_unbound = point_array.unbind();
        result_vector.push(
            PolygonResult {
                original_id: polygon.original_id,
                points: point_array_unbound
            }
        );
    }
    return PyList::new(py, result_vector);
}

pub fn test_tree<'py>(
    list: Vec<usize>
) -> PyResult<Vec<usize>> {
    let mut tree = double_key_tree::DoubleKeyTree::new();
    for item in 0..(list.len() / 2) {
        tree.insert_to_tree(list[item * 2], list[item * 2 + 1]);
    }
    Ok(tree.get_order())
}