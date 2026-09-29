use numpy::{PyReadonlyArray2};

pub fn numpy_to_vec<T: numpy::Element + Clone>(arr: &PyReadonlyArray2<'_, T>) -> Vec<T> {

    let data_and_size = arr
        .as_array()
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset();

    data_and_size.0
}