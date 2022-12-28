#include"sqlist.h"
int main() {
	SqList* L;
	ElemType e;
	int locate;
	cout << "初始化顺序表L" << endl;
	InitList(L);
	cout << "输入abcde" << endl;
	ListInsert(L, 1, 'a');
	ListInsert(L, 2, 'b');
	ListInsert(L, 3, 'c');
	ListInsert(L, 4, 'd');
	ListInsert(L, 5, 'e');
	cout << "输出L" << endl;;
	ListPrint(L);
	cout << "输出L长度" << endl;
	cout << ListLength(L) << endl;
	cout << "判断是否为空" << endl;
	if (ListEmpty(L))
		cout << "为空" << endl;
	else
		cout << "非空" << endl;
	cout << "输出第三个数据" << endl;
	if (ListGet(L, 3, e))
		cout << e << endl;
	else
		cout << "不存在" << endl;
	cout << "输出a的位置" << endl;
	if (locate = ListSearch(L, 'a'))
		cout << locate << endl;
	else
		cout << "不存在" << endl;
	cout << "在第四个位置插入f" << endl;
	ListInsert(L, 4, 'f');
	cout << "输出顺序表" << endl;
	ListPrint(L);
	cout << "删除第三个元素" << endl;
	ListDelete(L, 3, e);
	cout << "输出顺序表" << endl;
	ListPrint(L);
	cout << "释放顺序表" << endl;
	ListDestroy(L);

}