#include <stdio.h>
#include <stdint.h> // integer types, guaranteed sizes, as below
#include <stdbool.h> //bool type true or false
typedef int8_t i8; //8 bits signed (0 to positive)
typedef int16_t i16;
typedef int32_t i32; //most used  // negative 2147m to 2147m
typedef int64_t i64;
typedef uint8_t u8; //8 bits unsigned
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef i8 b8;
typedef i32 b32;

typedef float f32; 
typedef double f64;

typedef struct
{
    f32 x;
    f32 y;
} vec2f;

int main()
{
    i32 x = 123;

    i32* ppx = &x; //* can be at 2 or middle or before px; address corresponds to value of x (p)
    printf("%d %p\n", x, ppx);

    *ppx = 321;
    printf("%d %p\n", x, ppx);

    vec2f v = {1, 2};
    vec2f* pointerv = &v;

    printf("%f %f\n", v.x, v.y);

    return 0;
}
