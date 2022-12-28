#include"bst.h"

int main() {
	BSTNode* bt;
	int path[50];
	KeyType k = 6;
	int a[] = { 4,9,0,1,8,6,3,5,2,7 }, n = 10;
	cout << "³õÊ¼»¯Ò»¿Ã¶þ²æÅÅÐòÊ÷" << endl;
	bt = CreateBST(a, n);
	cout << "BST:";DispBST(bt);cout<<endl;
	cout << "ÅÐ¶ÏÊÇ·ñÎªÒ»¿Ã¶þ²æÅÅÐòÊ÷:";
	if (JudgeBST(bt))
		cout << "ÊÇÒ»¿Ã¶þ²æÅÅÐòÊ÷" << endl;
	else
		cout << "²»ÊÇÒ»¿Ã¶þ²æÅÅÐòÊ÷" << endl;
	cout << "²éÕÒ" << k << "¹Ø¼ü×Ö£¨·ÇµÝ¹é£¬Ë³Ðò£©";SearchBST1(bt, k, path,-1);
	cout << "²éÕÒ" << k << "¹Ø¼ü×Ö£¨µÝ¹é£¬ÄæÐò£©";SearchBST2(bt, k);
	cout << endl;
	cout << "É¾³ý½áµã" << endl;
	cout << "Ô­BST:";DispBST(bt);cout << endl;
	cout << "É¾³ý½áµã4:";DeleteBST(bt,4);DispBST(bt);cout << endl;
	cout << "É¾³ý½áµã5:";DeleteBST(bt, 5);DispBST(bt);cout << endl;
	cout << "Ïú»ÙBST:";DestoryBST(bt);
	cout << endl;
}