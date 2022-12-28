KeyType predt = -32768; //predt为全局变量,保存当前结点中序前驱的值,初值为-∞
bool JudgeBST(BSTNode* bt)
{
	bool b1, b2;
	if (bt == NULL)
		return true;
	else
	{
		b1 = JudgeBST(bt->lchild); //判断左子树
		if (b1 == false) //左子树不是BST，返回假
			return false;
		if (bt->key < predt) //当前结点违反BST性质，返回假
			return false;
		predt = bt->key;
		b2 = JudgeBST(bt->rchild); //判断右子树
		return b2;
	}
}