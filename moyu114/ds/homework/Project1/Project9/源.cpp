KeyType predt = -32768; //predtΪȫ�ֱ���,���浱ǰ�������ǰ����ֵ,��ֵΪ-��
bool JudgeBST(BSTNode* bt)
{
	bool b1, b2;
	if (bt == NULL)
		return true;
	else
	{
		b1 = JudgeBST(bt->lchild); //�ж�������
		if (b1 == false) //����������BST�����ؼ�
			return false;
		if (bt->key < predt) //��ǰ���Υ��BST���ʣ����ؼ�
			return false;
		predt = bt->key;
		b2 = JudgeBST(bt->rchild); //�ж�������
		return b2;
	}
}