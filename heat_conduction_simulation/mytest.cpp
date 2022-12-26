#include <stdio.h>

struct mytest
{
    /* data */
    int A;
    static mytest** get_my_ptr(void){
        static mytest* B;
        printf("Func get_my_ptr B %x %x\n", B, (void*)&B);
        return &B;
    }

    void test(){
        mytest** b = get_my_ptr();
        printf("test b %x\n", (void*)b);
        *b = this;
        printf("test b %x\n", (void*)b);
    }
};

int main(void){
    mytest test;
    test.test();
}
