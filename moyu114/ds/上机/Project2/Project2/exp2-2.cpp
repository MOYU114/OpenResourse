#include"linklist.h"
int main() {
	LinkNode* L;
	ElemType e;
	int locate;
	cout << "��ʼ���������L" << endl;
	InitNode(L);
	cout << "β�巨����a,b,c,d,e" << endl;
	NodeInsert(L, 1, 'a'); 
	NodeInsert(L, 2, 'b');
	NodeInsert(L, 3, 'c');
	NodeInsert(L, 4, 'd');
	NodeInsert(L, 5, 'e');
	cout << "���L" << endl;;
	NodePrint(L);
	cout << "���L����" << endl;
	cout<<NodeLength(L)<< endl;
	cout << "�ж��Ƿ�Ϊ��" << endl;
	if (NodeEmpty(L))
		cout << "Ϊ��" << endl;
	else
		cout << "�ǿ�" << endl;
	cout << "�������������" << endl;
	if (NodeGet(L, 3, e))
		cout << e << endl;
	else
		cout << "������" << endl;
	cout << "���a��λ��" << endl;
	if (locate = NodeSearch(L, 'a'))
		cout << locate << endl;
	else
		cout << "������" << endl;
	cout << "�ڵ��ĸ�λ�ò���f" << endl;
	NodeInsert(L, 4, 'f');
	cout << "���������" << endl;
	NodePrint(L);
	cout << "ɾ��������Ԫ��" << endl;
	NodeDelete(L, 3, e);
	cout << "���������" << endl;
	NodePrint(L);
	cout << "�ͷŵ�����" << endl;
	NodeDestroy(L);

}