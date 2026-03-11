use std::mem::swap;

pub trait HasDepthLimit<L, E> {
    fn depth_limit(&self) -> Option<usize>;
    fn stack_err(&self, loc: L) -> E;
}

pub trait Unvisit<T> {
    fn give_back(&mut self, info: Option<Box<T>>);
    fn take(&mut self) -> Option<Box<T>>;
    fn depth(&self) -> usize;
}

pub struct VisitedMarker<'info, T> {
    pub info: Option<Box<T>>,
    pub prev: Option<&'info mut dyn Unvisit<T>>,
    pub depth: usize,
}

impl<'info, T> VisitedMarker<'info, T> {
    pub fn new(info: T) -> VisitedMarker<'static, T> {
        VisitedMarker {
            info: Some(Box::new(info)),
            prev: None,
            depth: 1,
        }
    }

    // Each new level takes the info box and adds one depth.
    pub fn again<L, E>(
        loc: L,
        prev: &'info mut dyn Unvisit<T>,
    ) -> Result<VisitedMarker<'info, T>, E>
    where
        T: HasDepthLimit<L, E>,
    {
        let info = prev.take();
        let depth = prev.depth();
        if let Some(ref info) = info {
            if let Some(limit) = info.depth_limit() {
                if depth >= limit {
                    return Err(info.stack_err(loc));
                }
            }
        }
        Ok(VisitedMarker {
            info,
            prev: Some(prev),
            depth: depth + 1,
        })
    }
}

impl<T> Unvisit<T> for VisitedMarker<'_, T> {
    fn give_back(&mut self, info: Option<Box<T>>) {
        self.info = info;
    }
    fn take(&mut self) -> Option<Box<T>> {
        let mut info = None;
        swap(&mut self.info, &mut info);
        info
    }
    fn depth(&self) -> usize {
        self.depth
    }
}

// When dropped, the info box is handed back.
impl<T> Drop for VisitedMarker<'_, T> {
    fn drop(&mut self) {
        let mut info = None;
        swap(&mut self.info, &mut info);
        if let Some(ref mut prev) = self.prev {
            prev.give_back(info);
        }
    }
}
