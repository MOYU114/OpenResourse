#include"sqlist.h"
int main() {
	SqList* L;
	ElemType e;
	int locate;
	cout << "��ʼ��˳���L" << endl;
	InitList(L);
	cout << "����abcde" << endl;
	ListInsert(L, 1, 'a');
	ListInsert(L, 2, 'b');
	ListInsert(L, 3, 'c');
	ListInsert(L, 4, 'd');
	ListInsert(L, 5, 'e');
	cout << "���L" << endl;;
	ListPrint(L);
	cout << "���L����" << endl;
	cout << ListLength(L) << endl;
	cout << "�ж��Ƿ�Ϊ��" << endl;
	if (ListEmpty(L))
		cout << "Ϊ��" << endl;
	else
		cout << "�ǿ�" << endl;
	cout << "�������������" << endl;
	if (ListGet(L, 3, e))
		cout << e << endl;
	else
		cout << "������" << endl;
	cout << "���a��λ��" << endl;
	if (locate = ListSearch(L, 'a'))
		cout << locate << endl;
	else
		cout << "������" << endl;
	cout << "�ڵ��ĸ�λ�ò���f" << endl;
	ListInsert(L, 4, 'f');
	cout << "���˳���" << endl;
	ListPrint(L);
	cout << "ɾ��������Ԫ��" << endl;
	ListDelete(L, 3, e);
	cout << "���˳���" << endl;
	ListPrint(L);
	cout << "�ͷ�˳���" << endl;
	ListDestroy(L);

}