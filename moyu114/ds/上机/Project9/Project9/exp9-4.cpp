#include"bst.h"

int main() {
	BSTNode* bt;
	int path[50];
	KeyType k = 6;
	int a[] = { 4,9,0,1,8,6,3,5,2,7 }, n = 10;
	cout << "��ʼ��һ�ö���������" << endl;
	bt = CreateBST(a, n);
	cout << "BST:";DispBST(bt);cout<<endl;
	cout << "�ж��Ƿ�Ϊһ�ö���������:";
	if (JudgeBST(bt))
		cout << "��һ�ö���������" << endl;
	else
		cout << "����һ�ö���������" << endl;
	cout << "����" << k << "�ؼ��֣��ǵݹ飬˳��";SearchBST1(bt, k, path,-1);
	cout << "����" << k << "�ؼ��֣��ݹ飬����";SearchBST2(bt, k);
	cout << endl;
	cout << "ɾ�����" << endl;
	cout << "ԭBST:";DispBST(bt);cout << endl;
	cout << "ɾ�����4:";DeleteBST(bt,4);DispBST(bt);cout << endl;
	cout << "ɾ�����5:";DeleteBST(bt, 5);DispBST(bt);cout << endl;
	cout << "����BST:";DestoryBST(bt);
	cout << endl;
}