
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::types::PyDict;
use numpy::{PyArray, Ix2, PyArrayMethods, PyUntypedArrayMethods, PyReadonlyArray2};

fn numpy_to_vec(arr: &PyReadonlyArray2<'_, i64>) -> Vec<i64> {

    let data_and_size = arr
        .as_array()
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset();

    data_and_size.0
}

fn fill_point_neighbor(
    region_grid: &Vec<i32>, 
    result_dict: &Bound<'_, PyDict>, 
    y: usize, 
    x: usize, 
    image_size: &Vec<usize>,
    py: Python,
    original_region: i32
) {
    if y < image_size[0] && x < image_size[1] {
        let new_value = region_grid[y * image_size[1] + x];
        if new_value != original_region {
            if !result_dict.contains(original_region).expect("Region key check failted") {
                let region_list_insert = PyList::empty(py);
                result_dict.set_item(original_region, region_list_insert).expect("Region point list init failed");
            }
            let region_list_obj = result_dict.get_item(original_region).expect("List retrieval failed")
                .expect("List retrieval failed");
            let region_list = region_list_obj.cast::<PyList>().expect("List cast failed");
            let neighbor_point_list = PyList::new(py, [y, x]).expect("Neighbor list creation failed");
            region_list.append(neighbor_point_list).expect("Neighbor list insertion failed");
        }
    }
}

fn fill_point_neighbors(region_grid: &Vec<i32>, result_dict: &Bound<'_, PyDict>, y: usize, x: usize, image_size: &Vec<usize>, py: Python) {
    let original_region_id = region_grid[y * image_size[1] + x];
    fill_point_neighbor(&region_grid, &result_dict, y + 1, x, &image_size, py, original_region_id);
    fill_point_neighbor(&region_grid, &result_dict, y, x + 1, &image_size, py, original_region_id);
    if y > 0 {
        fill_point_neighbor(&region_grid, &result_dict, y - 1, x, &image_size, py, original_region_id);
    }
    if x > 0 {
        fill_point_neighbor(&region_grid, &result_dict, y, x - 1, &image_size, py, original_region_id);
    }
}

fn fill_region_grid<'py>(image_size: &Vec<usize>, region_points: Bound<'py, PyList>) -> PyResult<Vec<i32>> {
    let grid_elements = image_size[0] * image_size[1];
    let mut region_grid = vec![-1; grid_elements];
    for (region_id, region_array_obj) in region_points.iter().enumerate() {
      let region_array = region_array_obj.cast::<PyArray<i64, Ix2>>()
      .expect("Unable to cast region point array").readonly();
      let point_count = *region_array.shape().get(1).ok_or("No point count!").
        map_err(|e| PyValueError::new_err(e.to_string()))?;
      let region_vect = numpy_to_vec(&region_array);
      for point_index in 0..point_count {
        let point_y = region_vect[point_index];
        let point_x = region_vect[point_index + point_count];
        let region_grid_index = (point_y as usize) * image_size[1] + (point_x as usize);
        region_grid[region_grid_index] = region_id as i32;
      }
    }
    Ok(region_grid)
}


pub fn get_borders_dict<'py>(
    py: Python<'py>,
    image_size: Vec<usize>,
    region_points: Bound<'py, PyList>
  ) -> PyResult<Bound<'py, PyDict>> {
    let result_dict = PyDict::new(py);
    let region_grid = fill_region_grid(&image_size, region_points)?;
    for region_y in 0..image_size[0] {
      for region_x in 0..image_size[1] {
        fill_point_neighbors(&region_grid, &result_dict, region_y, region_x, &image_size, py);
      }
    }
    Ok(result_dict)
}

fn fill_double_point(
    result_dict: &Bound<'_, PyDict>, 
    y: usize, 
    x: usize, 
    py: Python,
    original_region: i64,
    new_region: i64
) {
    if !result_dict.contains(original_region).expect("Region key check failted") {
        let region_dict_insert = PyDict::new(py);
        result_dict.set_item(original_region, region_dict_insert).expect("Region point list init failed");
    }
    let region_dict_obj = result_dict.get_item(original_region).expect("List retrieval failed")
        .expect("List retrieval failed");
    let region_dict = region_dict_obj.cast::<PyDict>().expect("List cast failed");
    if !region_dict.contains(new_region).expect("Region key check failted") {
        let region_list_insert = PyList::empty(py);
        region_dict.set_item(new_region, region_list_insert).expect("Region point list init failed");
    }
    let region_list_obj = region_dict.get_item(new_region).expect("List retrieval failed")
        .expect("List retrieval failed");
    let region_list = region_list_obj.cast::<PyList>().expect("List cast failed");
    let neighbor_point_list = PyList::new(py, [y, x]).expect("Neighbor list creation failed");
    region_list.append(neighbor_point_list).expect("Neighbor list insertion failed");
}


fn fill_double_point_neighbor(
    region_grid: &Vec<i64>, 
    values_grid: &Vec<i64>, 
    result_dict: &Bound<'_, PyDict>, 
    y: usize, 
    x: usize, 
    image_size: &Vec<usize>,
    py: Python,
    original_region: i64,
    original_value: i64,
) {
    if y < image_size[0] && x < image_size[1] {
        let new_region_id = region_grid[y * image_size[1] + x];
        let new_value = values_grid[y * image_size[1] + x];
        let is_new_past_old = new_region_id > original_region;
        let are_regions_opaque= new_value >= 0 && original_value >= 0;
        let is_only_old_alpha = new_value != -1 && original_value == -1;
        if  is_new_past_old && are_regions_opaque {
            // if new_region_id == 1 && original_region == 1 {
            //     println!("Screwy path 1")
            // }
            fill_double_point(result_dict, y, x, py, original_region, new_region_id);
        }
        else if is_only_old_alpha {
            // if new_region_id == 1 && original_region == 1 {
            //     println!("Screwy path 2")
            // }
            fill_double_point(result_dict, y, x, py, new_region_id, original_region);
        }
    }
}

fn fill_double_point_neighbors(
        region_grid: &Vec<i64>, 
        values_grid: &Vec<i64>, 
        result_dict: &Bound<'_, PyDict>, 
        y: usize, 
        x: usize, 
        image_size: &Vec<usize>, 
        py: Python
    ) {
    let original_region_id = region_grid[y * image_size[1] + x];
    let original_value = values_grid[y * image_size[1] + x];
    fill_double_point_neighbor(
        region_grid, values_grid, result_dict, y + 1, x, &image_size, py, original_region_id, original_value
    );
    fill_double_point_neighbor(
        region_grid, values_grid, result_dict, y, x + 1, &image_size, py, original_region_id, original_value
    );
    if y > 0 {
        fill_double_point_neighbor(
            region_grid, values_grid, result_dict,
            y - 1, x, &image_size, py, original_region_id, original_value
        );
    }
    if x > 0 {
        fill_double_point_neighbor(
            region_grid, values_grid, result_dict,
            y, x - 1, &image_size, py, original_region_id, original_value
        );
    }
    if y == 0 || x == 0 {
        fill_double_point(
            result_dict, y, x, py, original_region_id, -1
        );
    }
}


pub fn get_double_border_dict<'py>(
    py: Python<'py>,
    image_size: Vec<usize>,
    region_map: Bound<'py, PyArray<i64, Ix2>>,
    point_values: Bound<'py, PyArray<i64, Ix2>>
  ) -> PyResult<Bound<'py, PyDict>> {
    let result_dict = PyDict::new(py);
    let region_grid = numpy_to_vec(&region_map.readonly());
    let values_grid = numpy_to_vec(&point_values.readonly());
    for region_y in 0..image_size[0] {
      for region_x in 0..image_size[1] {
        fill_double_point_neighbors(
            &region_grid, 
            &values_grid,
            &result_dict, 
            region_y, 
            region_x, 
            &image_size, 
            py
        );
      }
    }
    Ok(result_dict)
  }