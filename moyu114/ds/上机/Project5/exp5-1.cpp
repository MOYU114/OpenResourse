#include<iostream>
#include<malloc.h>
using namespace std;
typedef struct {
	int n;
	char x, y, z;
	bool flag;
}ElemType;
#define MAXSIZE 100
typedef struct {
	ElemType data[MAXSIZE];
	int top;
}Stacktype;

void InitStack(Stacktype*& s);
bool StackEmpty(Stacktype* s);
bool Push(Stacktype*& s, ElemType e);
bool Pop(Stacktype*& s, ElemType &e);

void DestoryStack(Stacktype*& s);
void Hanoi1(char a,char b,char c,int n) {
	if (n == 1)
		cout << "把第" << n << "个圆盘从" << a << "移动到" << c << endl;
	else {
		Hanoi1(a, c, b, n - 1);
		cout << "把第" << n << "个圆盘从" << a << "移动到" << c << endl;
		Hanoi1(b, a, c, n - 1);
	}
}
void Hanoi2(char x,char y,char z,int n) {
	Stacktype* s;
	ElemType e, e1, e2, e3;
	if (n <= 0)return;
	InitStack(s);
	e.n = n;e.x = x;e.y = y;e.z = z;e.flag = false;
	Push(s, e);
	while (!StackEmpty(s)) {
		Pop(s, e);
		if (e.flag == false) {
			e1.n = e.n-1; e1.x = e.y; e1.y = e.x; e1.z = e.z;
			if (e1.n == 1)
				e1.flag = true;
			else
				e1.flag = false;
			Push(s, e1);
			e2.n = e.n; e2.x = e.x; e2.y = e.y; e2.z = e.z; e2.flag = true;
			Push(s, e2);
			e3.n = e.n - 1;e3.x = e.x;e3.y = e.z;e3.z = e.y;
			if (e3.n == 1)
				e3.flag = true;
			else
				e3.flag = false;
			Push(s, e3);
		}
		else {
			cout<< "把第" << e.n << "个圆盘从" << e.x << "移动到" << e.z << endl;
		}
	}
	DestoryStack(s);
}
int main() {
	int n=3;
	Hanoi1('a', 'b', 'c', n);
	cout << endl;
	Hanoi2('a', 'b', 'c', n);
}
void InitStack(Stacktype*& s) {
	s = (Stacktype*)malloc(sizeof(Stacktype));
	s->top = -1;
}

bool StackEmpty(Stacktype* s) {
	return(s->top == -1);

}
bool Push(Stacktype*& s, ElemType e) {
	if (s->top == MAXSIZE - 1) {
		return false;
	}
	s->top++;
	s->data[s->top] = e;
	return true;
}
bool Pop(Stacktype*& s, ElemType& e)
{
	if (s->top == -1) {
		return false;
	}
	
		e = s->data[s->top];
		s->top--;
		return true;
	
}

void DestoryStack(Stacktype*& s) {
	free(s);
}