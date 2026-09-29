/// A Python module implemented in Rust. The name of this module must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
mod border_finder;
mod triangle_cutter;
mod double_key_tree;
mod numpy_utils;
mod mesh;
mod find_index_in;
mod triangle_cutter_python_interface;
#[pyo3::pymodule]
mod gradengrs {
  use pyo3::prelude::*;
  use pyo3::types::PyList;
  use pyo3::types::PyDict;
  use numpy::{PyArray, Ix2};
  use crate::triangle_cutter_python_interface;
  use crate::triangle_cutter;
  #[pymodule_export]
  use crate::triangle_cutter_python_interface::PolygonResult;

  #[pyfunction]
  fn get_borders_dict<'py>(
    py: Python<'py>,
    image_size: Vec<usize>,
    region_points: Bound<'py, PyList>
  ) -> PyResult<Bound<'py, PyDict>> {
    super::border_finder::get_borders_dict(py, image_size, region_points)
  }

  #[pyfunction]
  fn get_double_borders_dict<'py>(
    py: Python<'py>,
    image_size: Vec<usize>,
    region_map: Bound<'py, PyArray<i64, Ix2>>,
    point_values: Bound<'py, PyArray<i64, Ix2>>
  ) -> PyResult<Bound<'py, PyDict>> {
    super::border_finder::get_double_border_dict(py, image_size, region_map, point_values)
  }

  #[pyfunction]
  fn cut_out_trianges<'py> (
    py: Python<'py>,
    polygons: Bound<'py, PyList>
  ) -> PyResult<Bound<'py, PyList>> {
    let polygon_vector = triangle_cutter_python_interface::get_polygon_vector(polygons)?;
    let (rectancles, vertices) = triangle_cutter::cut_out_trianges(polygon_vector);
    triangle_cutter_python_interface::get_polygon_results(py, rectancles, vertices)
  }

  #[pyfunction]
  fn test_tree<'py>(
    list: Vec<usize>
  ) -> PyResult<Vec<usize>> {
    super::triangle_cutter_python_interface::test_tree(list)
  }
}