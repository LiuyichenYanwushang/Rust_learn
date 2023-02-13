// 红黑树的结构
struct RedBlackTree<T> {
    root: Option<Box<Node<T>>>,
    size: usize,
}

// 红黑树的节点
struct Node<T> {
    value: T,
    color: Color,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
    parent: Option<Box<Node<T>>>,
}

// 红黑树节点的颜色
enum Color {
    Red,
    Black,
}

impl<T> RedBlackTree<T> {
    // 插入节点
    fn insert(&mut self, value: T) {
        let mut new_node = Box::new(Node {
            value,
            color: Color::Red,
            left: None,
            right: None,
            parent: None,
        });
        // 找到合适的插入位置
        let mut curr = self.root.take();
        let mut parent = None;
        while let Some(node) = curr {
            parent = Some(node.clone());
            if new_node.value < node.value {
                curr = node.left.take();
            } else {
                curr = node.right.take();
            }
        }
        new_node.parent = parent;
        if let Some(mut parent) = parent {
            if new_node.value < parent.value {
                parent.left = Some(new_node);
            } else {
                parent.right = Some(new_node);
            }
        } else {
            self.root = Some(new_node);
        }

struct Node {
    key: i32,
    color: bool,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    parent: Option<Box<Node>>,
}

impl Node {
    fn new(key: i32) -> Self {
        Node {
            key,
            color: true,
            left: None,
            right: None,
            parent: None,
        }
    }
}

struct RBTree {
    root: Option<Box<Node>>,
}

impl RBTree {
    fn new() -> Self {
        RBTree { root: None }
    }

    fn insert(&mut self, key: i32) {
        let new_node = Box::new(Node::new(key));

        if self.root.is_none() {
            self.root = Some(new_node);
            return;
        }

        let mut current_node = &mut self.root;
        loop {
            if key < current_node.as_ref().unwrap().key {
                if current_node.as_mut().unwrap().left.is_none() {
                    current_node.as_mut().unwrap().left = Some(new_node);
                    break;
                } else {
                    current_node = &mut current_node.as_mut().unwrap().left;
                }
            } else {
                if current_node.as_mut().unwrap().right.is_none() {
                    current_node.as_mut().unwrap().right = Some(new_node);
                    break;
                } else {
                    current_node = &mut current_node.as_mut().unwrap().right;
                }
            }
        }
    }
}

main() {

}
