use std::cell::RefCell;
use std::rc::Weak;
use std::{rc::Rc};

use crate::double_key_tree::Direction::{LEFT, RIGHT};

struct DoubleKeyTreeNode {
    key_1: usize,
    key_2: usize,
    under_leaf: Option<Rc<RefCell<DoubleKeyTreeNode>>>,
    over_leaf: Option<Rc<RefCell<DoubleKeyTreeNode>>>,
    parent: Option<Weak<RefCell<DoubleKeyTreeNode>>>,
    color: Color
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    RIGHT,
    LEFT
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    RED,
    BLACK
}

impl DoubleKeyTreeNode {
    fn get_child(&self, 
        dir: Direction) -> Option<Rc<RefCell<DoubleKeyTreeNode>>>{
        match dir {
            LEFT => self.under_leaf.clone(),
            RIGHT => self.over_leaf.clone()
        }
    }
    fn new(key_1: usize, key_2: usize, parent: Option<Weak<RefCell<DoubleKeyTreeNode>>>) -> Self {
        DoubleKeyTreeNode {
            key_1, 
            key_2, 
            under_leaf: None, 
            over_leaf: None, 
            parent, 
            color: Color::RED
        }
    }
    
    fn fill_results(&self, fill_data: &mut Vec<usize>) {
        if let Some(some_underleaf) = self.under_leaf.clone() {
            some_underleaf.borrow().fill_results(fill_data);
        }
        fill_data.push(self.key_1);
        fill_data.push(self.key_2);
        if let Some(some_over_leaf) = self.over_leaf.clone() {
            some_over_leaf.borrow().fill_results(fill_data);
        }
    }
    
}

trait NodeInRef {
    fn set_child(&mut self, dir: Direction, new_child: Option<Rc<RefCell<DoubleKeyTreeNode>>>);
    fn rotate_subtree(&mut self, dir: Direction) -> Rc<RefCell<DoubleKeyTreeNode>>;
    fn is_color(&self, color: Color) -> bool;
    fn clone_parent(&self) -> Option<Weak<RefCell<DoubleKeyTreeNode>>>;
    fn set_color(&self, new_color: Color);
    fn direction(&self) -> Direction;
    fn direction_of(&self, key_1: usize, key_2: usize) -> bool;
    fn is_node_for(&self, key_1: usize, key_2: usize) -> bool;
    fn set_parent(&mut self, parent: Option<Weak<RefCell<DoubleKeyTreeNode>>>);
}


impl NodeInRef for Rc<RefCell<DoubleKeyTreeNode>> {
    fn set_child(&mut self, dir: Direction, new_child: Option<Rc<RefCell<DoubleKeyTreeNode>>>) {
        if let Some(mut some_new_child) = new_child.clone() {
            some_new_child.set_parent(Some(Rc::downgrade(&self)));
        }
        let previous_child;
        match dir {
            LEFT => {
                previous_child = self.borrow().get_child(LEFT);
                self.borrow_mut().under_leaf = new_child.clone();
            },
            RIGHT => {
                previous_child = self.borrow().get_child(RIGHT);
                self.borrow_mut().over_leaf = new_child.clone();
            }
        }
        if let Some(mut last_dir_child) = previous_child {
            if let Some(some_new_child) = new_child && 
                    !Rc::ptr_eq(&last_dir_child, &some_new_child) {
                last_dir_child.set_parent(None);
            }
        }
    }
    fn set_parent(&mut self, new_parent: Option<Weak<RefCell<DoubleKeyTreeNode>>>) {
        if let Some(parent) = self.clone_parent() {
            if let Some(mut strong_parent) = parent.upgrade() {
                if let Some(new_strong_parent) = new_parent.clone() && 
                    !Weak::ptr_eq(&new_strong_parent, &parent) {
                        strong_parent.set_child(self.direction(), None);
                }
            }
        }
        self.borrow_mut().parent = new_parent;
    }
    fn rotate_subtree(&mut self, dir: Direction) -> Rc<RefCell<DoubleKeyTreeNode>> {
        let sub_parent;
        let mut new_root;
        let new_child;
        {
            // let unwrapped_self = self.borrow();
            sub_parent = self.clone_parent();
            let new_root_wrapped_res = self.borrow()
                .get_child(match dir {LEFT => RIGHT, RIGHT => LEFT}); // 1 - dir is the opposite direction
            new_root = new_root_wrapped_res.expect("Called rotate_subtree on a node with no matching child");
            new_child = new_root.borrow()
                .get_child(dir);
        }

        self.set_child(match dir {LEFT => RIGHT, RIGHT => LEFT}, new_child.clone());

        if let Some(some_sub_parent) = sub_parent {
            if let Some(mut some_parent_strong) = some_sub_parent.upgrade() {
                let sub_parent_set_dir = self.direction();
                some_parent_strong.set_child(sub_parent_set_dir, Some(Rc::clone(&new_root)));
            }
        }
        else {
            new_root.set_parent(None);
        }
        new_root.set_child(dir, Some(Rc::clone(&self)));
        return new_root;
    }
    fn is_color(&self, color: Color) -> bool {
        self.borrow().color == color
    }
    fn clone_parent(&self) -> Option<Weak<RefCell<DoubleKeyTreeNode>>> {
        self.borrow().parent.clone()
    }
    fn set_color(&self, new_color: Color) {
        self.borrow_mut().color = new_color;
    }
    fn direction(&self) -> Direction {
        let some_parent_binding = self.clone_parent()
            .expect("Direction called on a node without a parent");
        match some_parent_binding.upgrade()
        .expect("Couln't upgrade reference to parent when finding direction")
        .borrow()
        .get_child(RIGHT) {
            Some(parent_overleaf) => if Rc::ptr_eq(&self, &parent_overleaf) {RIGHT} else {LEFT},
            None => LEFT
        }
    }

    fn direction_of(&self, key_1: usize, key_2: usize) -> bool {
        let self_bound = self.borrow();
        self_bound.key_1 > key_1 || (self_bound.key_1 == key_1 && self_bound.key_2 > key_2)
    }
    fn is_node_for(&self, key_1: usize, key_2: usize) -> bool {
        let self_bound = self.borrow();
        self_bound.key_1 == key_1 && self_bound.key_2 == key_2
    }
}


pub struct DoubleKeyTree {
    root: Option<Rc<RefCell<DoubleKeyTreeNode>>>,
    count: usize
}

impl DoubleKeyTree {
    
    pub fn insert_to_tree(&mut self, key_1: usize, key_2: usize) {
        let mut search_node = self.root.clone();
        loop {
            if let Some(some_search_node) = search_node.clone() {
                let node_to_left = some_search_node.direction_of(key_1, key_2);
                if node_to_left {
                    let search_node_underleaf = some_search_node.borrow().get_child(Direction::LEFT);
                    match search_node_underleaf {
                        Some(_) => search_node = search_node_underleaf,
                        None => break
                    }
                }
                else if some_search_node.is_node_for(key_1, key_2) {
                    break;
                }
                else {
                    let search_node_overleaf = some_search_node.borrow().get_child(Direction::RIGHT);
                    match search_node_overleaf {
                        Some(_) => search_node = search_node_overleaf,
                        None => break
                    }
                }
            }
            else {
                break;
            }
        }

        let node_parent;
        if let Some(some_search_node) = search_node.clone() {
            if some_search_node.is_node_for(key_1, key_2) {
                return;
            }

            node_parent = Some(Rc::downgrade(&some_search_node));
        }
        else {
            node_parent = None;
        }
        

        let mut node = Rc::new(RefCell::new(DoubleKeyTreeNode::new(
            key_1, 
            key_2, 
            node_parent
        )));
        self.count += 1;

        let Some(mut some_search_node) = search_node.clone() else {
            self.root = Some(Rc::clone(&node));
            return;
        };

        let mut dir;
        {
            let node_to_left = some_search_node.direction_of(key_1, key_2);
            dir = if node_to_left {Direction::LEFT} else {Direction::RIGHT};
        }
        some_search_node.set_child(dir, Some(Rc::clone(&node)));

        // rebalance the tree 
        loop {
            // Case #1
            if some_search_node.is_color(Color::BLACK) {
                break; 
            }

            let grandparent = some_search_node.clone_parent();

            let Some(some_grandparent) = grandparent else {
                // Case #4
                some_search_node.set_color(Color::BLACK);
                break;
            };
            
            let mut strong_grandparent = some_grandparent
                .upgrade()
                .expect("Couln't upgrade grandparent during insertion");

            dir = some_search_node.direction();
            let uncle = some_grandparent
                .upgrade()
                .expect("Couldn't upgrade grandparent reference when accessing uncle")
                .borrow()
                .get_child(match dir {LEFT => Direction::RIGHT, RIGHT => LEFT});


            let lacks_red_uncle = match uncle {
                Some(ref some_uncle) => some_uncle.is_color(Color::BLACK),
                None => true
            };
            if lacks_red_uncle {
                let sibling_result =  some_search_node.borrow().get_child(match dir {LEFT => Direction::RIGHT, RIGHT => LEFT});
                if let Some(some_sibling) = sibling_result {
                    if Rc::ptr_eq(&node, &some_sibling) {
                        // Case #5
                        let root_after_rotation = some_search_node.rotate_subtree(dir);
                        if root_after_rotation.clone_parent().is_none() {
                            self.root = Some(root_after_rotation);
                        }
                        some_search_node = strong_grandparent.borrow().get_child(dir).expect("Case 5 grandparent child get failed");
                    }
                }

                // Case #6
                let root_after_rotation = strong_grandparent.rotate_subtree( match dir {LEFT => Direction::RIGHT, RIGHT => LEFT});

                if root_after_rotation.clone_parent().is_none() {
                    self.root = Some(root_after_rotation);
                }
                some_search_node.set_color(Color::BLACK);
                strong_grandparent.set_color(Color::RED);
                break;
            }
        
            // Case #2
            some_search_node.set_color(Color::BLACK);
            if let Some(some_uncle) = uncle {
                some_uncle.set_color(Color::BLACK);
            }
            strong_grandparent.set_color(Color::RED);

            node = some_grandparent.upgrade().expect("Couldn't get strong reference to grandparent in Case 2");
            
            let new_search_node = node.clone_parent();
            if let Some(new_some_search_node) = new_search_node {
                if let Some(new_some_search_node_upgrade) = new_some_search_node.upgrade() {
                    some_search_node = new_some_search_node_upgrade;
                }
                else {
                    break;
                }
            } else {
                break;
            };
        }
        // This code is adapted from the red-black tree implementation provided in Wikipedia, which omits
        // the requirement stipulated by Cormen & al. that the root node remain black
        // node.set_color(BLACK)

        // Case #3
    }
    
    
    pub fn get_order(&self) -> Vec<usize> {
        let mut fill_data = Vec::with_capacity(self.count * 2);
        if let Some(ref some_root) = self.root {
            some_root.borrow().fill_results(&mut fill_data);
        }
        fill_data
    }

    pub fn new() -> DoubleKeyTree {
        return DoubleKeyTree {root: None, count: 0};
    }
}
