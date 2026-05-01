use rue_parser::{SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken, T};

pub trait AstNode {
    fn cast(node: SyntaxNode) -> Option<Self>
    where
        Self: Sized;

    fn syntax(&self) -> &SyntaxNode;
}

macro_rules! ast_node_impl {
    ($name:ident, $kind:ident) => {
        #[derive(Debug, Clone)]
        pub struct $name(SyntaxNode);

        impl AstNode for $name {
            fn cast(node: SyntaxNode) -> Option<Self> {
                match node.kind() {
                    SyntaxKind::$kind => Some(Self(node)),
                    _ => None,
                }
            }

            fn syntax(&self) -> &SyntaxNode {
                &self.0
            }
        }
    };
}

macro_rules! ast_enum_impl {
    ($name:ident, $( $variant:ident($node:ident) ),+ $(,)? ) => {
        #[derive(Debug, Clone)]
        pub enum $name {
            $( $variant($node), )+
        }

        impl AstNode for $name {
            fn cast(node: SyntaxNode) -> Option<Self> {
                $( if let Some(node) = $node::cast(node.clone()) {
                    return Some(Self::$variant(node));
                } )+
                None
            }

            fn syntax(&self) -> &SyntaxNode {
                match self {
                    $( Self::$variant(node) => node.syntax(), )+
                }
            }
        }
    };
}

ast_node_impl!(AstDocument, Document);
ast_node_impl!(AstModuleItem, ModuleItem);
ast_node_impl!(AstFunctionItem, FunctionItem);
ast_node_impl!(AstFunctionParameter, FunctionParameter);
ast_node_impl!(AstConstantItem, ConstantItem);
ast_node_impl!(AstTypeAliasItem, TypeAliasItem);
ast_node_impl!(AstStructItem, StructItem);
ast_node_impl!(AstStructField, StructField);
ast_node_impl!(AstImportItem, ImportItem);
ast_node_impl!(AstImportPath, ImportPath);
ast_node_impl!(AstImportPathSegment, ImportPathSegment);
ast_node_impl!(AstGenericParameters, GenericParameters);
ast_node_impl!(AstGenericArguments, GenericArguments);
ast_node_impl!(AstLiteralType, LiteralType);
ast_node_impl!(AstPathType, PathType);
ast_node_impl!(AstUnionType, UnionType);
ast_node_impl!(AstGroupType, GroupType);
ast_node_impl!(AstPairType, PairType);
ast_node_impl!(AstListType, ListType);
ast_node_impl!(AstListTypeItem, ListTypeItem);
ast_node_impl!(AstLambdaType, LambdaType);
ast_node_impl!(AstLambdaParameter, LambdaParameter);
ast_node_impl!(AstBlock, Block);
ast_node_impl!(AstLetStmt, LetStmt);
ast_node_impl!(AstExprStmt, ExprStmt);
ast_node_impl!(AstIfStmt, IfStmt);
ast_node_impl!(AstReturnStmt, ReturnStmt);
ast_node_impl!(AstAssertStmt, AssertStmt);
ast_node_impl!(AstRaiseStmt, RaiseStmt);
ast_node_impl!(AstDebugStmt, DebugStmt);
ast_node_impl!(AstPathExpr, PathExpr);
ast_node_impl!(AstPathSegment, PathSegment);
ast_node_impl!(AstStructInitializerExpr, StructInitializerExpr);
ast_node_impl!(AstStructInitializerField, StructInitializerField);
ast_node_impl!(AstLiteralExpr, LiteralExpr);
ast_node_impl!(AstGroupExpr, GroupExpr);
ast_node_impl!(AstPairExpr, PairExpr);
ast_node_impl!(AstListExpr, ListExpr);
ast_node_impl!(AstListItem, ListItem);
ast_node_impl!(AstPrefixExpr, PrefixExpr);
ast_node_impl!(AstBinaryExpr, BinaryExpr);
ast_node_impl!(AstFunctionCallExpr, FunctionCallExpr);
ast_node_impl!(AstIfExpr, IfExpr);
ast_node_impl!(AstGuardExpr, GuardExpr);
ast_node_impl!(AstCastExpr, CastExpr);
ast_node_impl!(AstFieldAccessExpr, FieldAccessExpr);
ast_node_impl!(AstLambdaExpr, LambdaExpr);
ast_node_impl!(AstNamedBinding, NamedBinding);
ast_node_impl!(AstPairBinding, PairBinding);
ast_node_impl!(AstListBinding, ListBinding);
ast_node_impl!(AstListBindingItem, ListBindingItem);
ast_node_impl!(AstStructBinding, StructBinding);
ast_node_impl!(AstStructFieldBinding, StructFieldBinding);

ast_enum_impl!(
    AstItem,
    TypeItem(AstTypeItem),
    SymbolItem(AstSymbolItem),
    ImportItem(AstImportItem),
);
ast_enum_impl!(
    AstTypeItem,
    TypeAliasItem(AstTypeAliasItem),
    StructItem(AstStructItem),
);
ast_enum_impl!(
    AstSymbolItem,
    ModuleItem(AstModuleItem),
    FunctionItem(AstFunctionItem),
    ConstantItem(AstConstantItem),
);
ast_enum_impl!(
    AstStmt,
    LetStmt(AstLetStmt),
    ExprStmt(AstExprStmt),
    IfStmt(AstIfStmt),
    ReturnStmt(AstReturnStmt),
    AssertStmt(AstAssertStmt),
    RaiseStmt(AstRaiseStmt),
    DebugStmt(AstDebugStmt),
);
ast_enum_impl!(AstStmtOrExpr, Stmt(AstStmt), Expr(AstExpr));
ast_enum_impl!(
    AstExpr,
    PathExpr(AstPathExpr),
    StructInitializerExpr(AstStructInitializerExpr),
    LiteralExpr(AstLiteralExpr),
    GroupExpr(AstGroupExpr),
    PairExpr(AstPairExpr),
    ListExpr(AstListExpr),
    PrefixExpr(AstPrefixExpr),
    BinaryExpr(AstBinaryExpr),
    FunctionCallExpr(AstFunctionCallExpr),
    Block(AstBlock),
    IfExpr(AstIfExpr),
    GuardExpr(AstGuardExpr),
    CastExpr(AstCastExpr),
    FieldAccessExpr(AstFieldAccessExpr),
    LambdaExpr(AstLambdaExpr),
);
ast_enum_impl!(
    AstType,
    LiteralType(AstLiteralType),
    PathType(AstPathType),
    UnionType(AstUnionType),
    GroupType(AstGroupType),
    PairType(AstPairType),
    ListType(AstListType),
    LambdaType(AstLambdaType),
);
ast_enum_impl!(
    AstBinding,
    NamedBinding(AstNamedBinding),
    PairBinding(AstPairBinding),
    ListBinding(AstListBinding),
    StructBinding(AstStructBinding),
);

impl AstDocument {
    pub fn items(&self) -> impl Iterator<Item = AstItem> {
        self.syntax().children().filter_map(AstItem::cast)
    }
}

impl AstModuleItem {
    pub fn export(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![export])
    }

    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }

    pub fn items(&self) -> impl Iterator<Item = AstItem> {
        self.syntax().children().filter_map(AstItem::cast)
    }
}

impl AstFunctionItem {
    pub fn export(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![export])
    }

    pub fn extern_kw(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![extern])
    }

    pub fn inline(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![inline])
    }

    pub fn test(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![test])
    }

    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }

    pub fn generic_parameters(&self) -> Option<AstGenericParameters> {
        self.syntax()
            .children()
            .find_map(AstGenericParameters::cast)
    }

    pub fn parameters(&self) -> impl Iterator<Item = AstFunctionParameter> {
        self.syntax()
            .children()
            .filter_map(AstFunctionParameter::cast)
    }

    pub fn return_type(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }

    pub fn body(&self) -> Option<AstBlock> {
        self.syntax().children().find_map(AstBlock::cast)
    }
}

impl AstConstantItem {
    pub fn export(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![export])
    }

    pub fn inline(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![inline])
    }

    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }

    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }

    pub fn value(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstTypeAliasItem {
    pub fn export(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![export])
    }

    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }

    pub fn generic_parameters(&self) -> Option<AstGenericParameters> {
        self.syntax()
            .children()
            .find_map(AstGenericParameters::cast)
    }

    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }
}

impl AstStructItem {
    pub fn export(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![export])
    }

    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }

    pub fn generic_parameters(&self) -> Option<AstGenericParameters> {
        self.syntax()
            .children()
            .find_map(AstGenericParameters::cast)
    }

    pub fn fields(&self) -> impl Iterator<Item = AstStructField> {
        self.syntax().children().filter_map(AstStructField::cast)
    }
}

impl AstStructField {
    pub fn spread(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![...])
    }

    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }

    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }

    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstImportItem {
    pub fn export(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![export])
    }

    pub fn path(&self) -> Option<AstImportPath> {
        self.syntax().children().find_map(AstImportPath::cast)
    }
}

impl AstImportPath {
    pub fn segments(&self) -> impl Iterator<Item = AstImportPathSegment> {
        self.syntax()
            .children()
            .filter_map(AstImportPathSegment::cast)
    }
}

impl AstImportPathSegment {
    pub fn separator(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![::])
    }

    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident || token.kind() == T![super])
    }

    pub fn star(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![*])
    }

    pub fn items(&self) -> impl Iterator<Item = AstImportPath> {
        self.syntax().children().filter_map(AstImportPath::cast)
    }
}

impl AstFunctionParameter {
    pub fn spread(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![...])
    }

    pub fn binding(&self) -> Option<AstBinding> {
        self.syntax().children().find_map(AstBinding::cast)
    }

    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }
}

impl AstGenericParameters {
    pub fn names(&self) -> impl Iterator<Item = SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .filter(|token| token.kind() == SyntaxKind::Ident)
    }
}

impl AstGenericArguments {
    pub fn types(&self) -> impl Iterator<Item = AstType> {
        self.syntax().children().filter_map(AstType::cast)
    }
}

impl AstBlock {
    pub fn items(&self) -> impl Iterator<Item = AstStmtOrExpr> {
        self.syntax().children().filter_map(AstStmtOrExpr::cast)
    }
}

impl AstLetStmt {
    pub fn inline(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![inline])
    }

    pub fn binding(&self) -> Option<AstBinding> {
        self.syntax().children().find_map(AstBinding::cast)
    }

    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }

    pub fn value(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstExprStmt {
    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstIfStmt {
    pub fn inline(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![inline])
    }

    pub fn condition(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }

    pub fn then_block(&self) -> Option<AstBlock> {
        self.syntax()
            .children()
            .filter(|node| AstExpr::cast(node.clone()).is_some())
            .nth(1)
            .and_then(AstBlock::cast)
    }
}

impl AstReturnStmt {
    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstAssertStmt {
    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstRaiseStmt {
    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstDebugStmt {
    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstPathExpr {
    pub fn segments(&self) -> impl Iterator<Item = AstPathSegment> {
        self.syntax().children().filter_map(AstPathSegment::cast)
    }
}

impl AstPathSegment {
    pub fn separator(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![::])
            .filter(|token| {
                let Some(name) = self.name() else {
                    return true;
                };
                token.text_range().start() < name.text_range().start()
            })
    }

    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident || token.kind() == T![super])
    }

    pub fn generic_arguments(&self) -> Option<AstGenericArguments> {
        self.syntax().children().find_map(AstGenericArguments::cast)
    }
}

impl AstStructInitializerExpr {
    pub fn path(&self) -> Option<AstPathExpr> {
        self.syntax().children().find_map(AstPathExpr::cast)
    }

    pub fn fields(&self) -> impl Iterator<Item = AstStructInitializerField> {
        self.syntax()
            .children()
            .filter_map(AstStructInitializerField::cast)
    }
}

impl AstStructInitializerField {
    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }

    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstLiteralExpr {
    pub fn value(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| SyntaxKind::LITERAL.contains(&token.kind()))
    }
}

impl AstGroupExpr {
    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstPairExpr {
    pub fn first(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }

    pub fn rest(&self) -> Option<AstExpr> {
        self.syntax().children().filter_map(AstExpr::cast).nth(1)
    }
}

impl AstListExpr {
    pub fn items(&self) -> impl Iterator<Item = AstListItem> {
        self.syntax().children().filter_map(AstListItem::cast)
    }
}

impl AstListItem {
    pub fn spread(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![...])
    }

    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstPrefixExpr {
    pub fn op(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| SyntaxKind::PREFIX_OPS.contains(&token.kind()))
    }

    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstBinaryExpr {
    pub fn op(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| SyntaxKind::BINARY_OPS.contains(&token.kind()))
    }

    pub fn left(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }

    pub fn right(&self) -> Option<AstExpr> {
        self.syntax().children().filter_map(AstExpr::cast).nth(1)
    }
}

impl AstFunctionCallExpr {
    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }

    pub fn args(&self) -> impl Iterator<Item = AstListItem> {
        self.syntax().children().filter_map(AstListItem::cast)
    }
}

impl AstIfExpr {
    pub fn inline(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![inline])
    }

    pub fn condition(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }

    pub fn then_expr(&self) -> Option<AstExpr> {
        self.syntax().children().filter_map(AstExpr::cast).nth(1)
    }

    pub fn else_expr(&self) -> Option<AstExpr> {
        self.syntax().children().filter_map(AstExpr::cast).nth(2)
    }
}

impl AstGuardExpr {
    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }

    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }
}

impl AstCastExpr {
    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }

    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }
}

impl AstFieldAccessExpr {
    pub fn expr(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }

    pub fn dot(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![.])
    }

    pub fn field(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }
}

impl AstLambdaExpr {
    pub fn generic_parameters(&self) -> Option<AstGenericParameters> {
        self.syntax()
            .children()
            .find_map(AstGenericParameters::cast)
    }

    pub fn parameters(&self) -> impl Iterator<Item = AstFunctionParameter> {
        self.syntax()
            .children()
            .filter_map(AstFunctionParameter::cast)
    }

    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }

    pub fn body(&self) -> Option<AstExpr> {
        self.syntax().children().find_map(AstExpr::cast)
    }
}

impl AstLiteralType {
    pub fn value(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| SyntaxKind::LITERAL.contains(&token.kind()))
    }
}

impl AstPathType {
    pub fn segments(&self) -> impl Iterator<Item = AstPathSegment> {
        self.syntax().children().filter_map(AstPathSegment::cast)
    }
}

impl AstUnionType {
    pub fn types(&self) -> impl Iterator<Item = AstType> {
        self.syntax().children().filter_map(AstType::cast)
    }
}

impl AstGroupType {
    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }
}

impl AstPairType {
    pub fn first(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }

    pub fn rest(&self) -> Option<AstType> {
        self.syntax().children().filter_map(AstType::cast).nth(1)
    }
}

impl AstListType {
    pub fn items(&self) -> impl Iterator<Item = AstListTypeItem> {
        self.syntax().children().filter_map(AstListTypeItem::cast)
    }
}

impl AstListTypeItem {
    pub fn spread(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![...])
    }

    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }
}

impl AstLambdaType {
    pub fn generic_parameters(&self) -> Option<AstGenericParameters> {
        self.syntax()
            .children()
            .find_map(AstGenericParameters::cast)
    }

    pub fn parameters(&self) -> impl Iterator<Item = AstLambdaParameter> {
        self.syntax()
            .children()
            .filter_map(AstLambdaParameter::cast)
    }

    pub fn return_type(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }
}

impl AstLambdaParameter {
    pub fn spread(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![...])
    }

    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }

    pub fn ty(&self) -> Option<AstType> {
        self.syntax().children().find_map(AstType::cast)
    }
}

impl AstNamedBinding {
    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }
}

impl AstPairBinding {
    pub fn first(&self) -> Option<AstBinding> {
        self.syntax().children().find_map(AstBinding::cast)
    }

    pub fn rest(&self) -> Option<AstBinding> {
        self.syntax().children().filter_map(AstBinding::cast).nth(1)
    }
}

impl AstListBinding {
    pub fn items(&self) -> impl Iterator<Item = AstListBindingItem> {
        self.syntax()
            .children()
            .filter_map(AstListBindingItem::cast)
    }
}

impl AstListBindingItem {
    pub fn spread(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![...])
    }

    pub fn binding(&self) -> Option<AstBinding> {
        self.syntax().children().find_map(AstBinding::cast)
    }
}

impl AstStructBinding {
    pub fn fields(&self) -> impl Iterator<Item = AstStructFieldBinding> {
        self.syntax()
            .children()
            .filter_map(AstStructFieldBinding::cast)
    }
}

impl AstStructFieldBinding {
    pub fn spread(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == T![...])
    }

    pub fn name(&self) -> Option<SyntaxToken> {
        self.syntax()
            .children_with_tokens()
            .filter_map(SyntaxElement::into_token)
            .find(|token| token.kind() == SyntaxKind::Ident)
    }

    pub fn binding(&self) -> Option<AstBinding> {
        self.syntax().children().find_map(AstBinding::cast)
    }
}
