/// A minimal no_std compatible monadic do-notation macro.
/// This replaces the `do-notation` crate's `m!` macro.
#[macro_export]
macro_rules! m {
    // Monadic bind: ident <- expr ; rest - MUST BE FIRST (before expr rule)
    { $name:ident <- $($rest:tt)+ } => {
        m!(@bind $name () $($rest)+)
    };

    // Let binding followed by more statements
    { let $p:pat = $v:expr ; $($rest:tt)+ } => {{
        let $p = $v;
        m!{ $($rest)+ }
    }};

    // Base case: final expression (must be AFTER bind rule)
    { $e:expr } => { $e };

    // Internal: accumulate expression tokens until semicolon
    (@bind $name:ident ($($acc:tt)*) ; $($rest:tt)+) => {
        ($($acc)*).and_then(|$name| m!{ $($rest)+ })
    };

    // Internal: keep accumulating
    (@bind $name:ident ($($acc:tt)*) $tok:tt $($rest:tt)*) => {
        m!(@bind $name ($($acc)* $tok) $($rest)*)
    };
}
