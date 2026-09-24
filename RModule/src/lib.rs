/// A Python module implemented in Rust. The name of this module must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
mod border_finder;
#[pyo3::pymodule]
mod gradengrs {
  use pyo3::prelude::*;
  use pyo3::types::PyList;
  use pyo3::types::PyDict;
  use numpy::{PyArray, Ix2};

  /// Formats the sum of two numbers as string.
  #[pyfunction]
  fn get_borders_dict<'py>(
    py: Python<'py>,
    image_size: Vec<usize>,
    region_points: Bound<'py, PyList>
  ) -> PyResult<Bound<'py, PyDict>> {
    super::border_finder::get_borders_dict(py, image_size, region_points)
  }

  /// Formats the sum of two numbers as string.
  #[pyfunction]
  fn get_double_borders_dict<'py>(
    py: Python<'py>,
    image_size: Vec<usize>,
    region_map: Bound<'py, PyArray<i64, Ix2>>,
    point_values: Bound<'py, PyArray<i64, Ix2>>
  ) -> PyResult<Bound<'py, PyDict>> {
    super::border_finder::get_double_border_dict(py, image_size, region_map, point_values)
  }
}