use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Less, Greater};
use std::fmt::Display;

pub trait BinaryKeyOrd {
    fn search_on(&self, key_1: usize, key_2: usize) -> Ordering;
}



pub fn find_index_in<T: BinaryKeyOrd>(
    key_1: usize,
    key_2: usize,
    orderable: &Vec<T>
) -> Option<usize> {
    assert_ne!(orderable.len(), 0);
    // Finds the location of the specified coordinates within a sorted array of points
    let mut lower_bound: usize = 0;
    let mut upper_bound: usize = orderable.len() - 1;
    let mut last_span = upper_bound - lower_bound;
    while lower_bound <= upper_bound {
        let midpoint = (lower_bound + upper_bound) / 2;
        let midpoint_ord = orderable[midpoint].search_on(key_1, key_2);
        match midpoint_ord {
            Equal => return Some(midpoint),
            Less => {
                if midpoint == upper_bound {return None;} else {lower_bound = midpoint + 1}
            },
            Greater => {
                if midpoint == 0 {return None;} else {upper_bound = midpoint - 1}
            }
        }
        assert!(last_span > upper_bound - lower_bound);
        last_span = upper_bound - lower_bound;
    }
    None
}


pub fn insert_into_double_key_ordered<T: BinaryKeyOrd + Display, F: FnOnce() -> T>(
    key_1: usize,
    key_2: usize,
    orderable: &mut Vec<T>,
    instantiator: F
) {
    // Updates a sorted list of elements with a new element with the following keys, if a matching element does not already exist
    if orderable.len() == 0 {
        orderable.insert(0, instantiator());
    }
    else if orderable.len() == 1 {
        match orderable[0].search_on(key_1, key_2) {
            Equal => return,
            Less => orderable.insert(0, instantiator()),
            Greater => orderable.insert(1, instantiator())
        }
    }
    else {
        let mut lower_bound: usize = 0;
        let mut upper_bound: usize = orderable.len() - 1;
        let mut last_span = upper_bound - lower_bound;
        while (lower_bound + 1) < upper_bound {
            let midpoint = (lower_bound + upper_bound) / 2;
            let midpoint_ord = orderable[midpoint].search_on(key_1, key_2);
            match midpoint_ord {
                Equal => return,
                Less => lower_bound = midpoint,
                Greater => upper_bound = midpoint
            }
            assert!(last_span > upper_bound - lower_bound);
            last_span = upper_bound - lower_bound;
        }
        let lower_orderable = &orderable[lower_bound];
        let upper_orderable = &orderable[upper_bound];
        let lower_ordering = lower_orderable.search_on(key_1, key_2);
        let upper_ordering = upper_orderable.search_on(key_1, key_2);
        match lower_ordering {
            Equal => return,
            Less => {},
            Greater => {
                orderable.insert(lower_bound, instantiator());
                return;
            }
        }
        match upper_ordering {
            Equal => return,
            Less => {
                orderable.insert(upper_bound + 1, instantiator());
                return;
            },
            Greater => {}
        }
        orderable.insert(upper_bound, instantiator());
    }
}
