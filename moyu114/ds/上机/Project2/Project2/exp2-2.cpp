#include"linklist.h"
int main() {
	LinkNode* L;
	ElemType e;
	int locate;
	cout << "初始化单链表表L" << endl;
	InitNode(L);
	cout << "尾插法插入a,b,c,d,e" << endl;
	NodeInsert(L, 1, 'a'); 
	NodeInsert(L, 2, 'b');
	NodeInsert(L, 3, 'c');
	NodeInsert(L, 4, 'd');
	NodeInsert(L, 5, 'e');
	cout << "输出L" << endl;;
	NodePrint(L);
	cout << "输出L长度" << endl;
	cout<<NodeLength(L)<< endl;
	cout << "判断是否为空" << endl;
	if (NodeEmpty(L))
		cout << "为空" << endl;
	else
		cout << "非空" << endl;
	cout << "输出第三个数据" << endl;
	if (NodeGet(L, 3, e))
		cout << e << endl;
	else
		cout << "不存在" << endl;
	cout << "输出a的位置" << endl;
	if (locate = NodeSearch(L, 'a'))
		cout << locate << endl;
	else
		cout << "不存在" << endl;
	cout << "在第四个位置插入f" << endl;
	NodeInsert(L, 4, 'f');
	cout << "输出单链表" << endl;
	NodePrint(L);
	cout << "删除第三个元素" << endl;
	NodeDelete(L, 3, e);
	cout << "输出单链表" << endl;
	NodePrint(L);
	cout << "释放单链表" << endl;
	NodeDestroy(L);

}