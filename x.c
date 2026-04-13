#include <stdio.h>
#include <stdint.h> // integer types, guaranteed sizes, as below
#include <stdbool.h> //bool type true or false
typedef int8_t i8; //8 bits signed (0 to positive)
typedef int16_t i6;
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

struct vec2f //grouping of data, plain data // 4 bytes on x and  4 on y cuz typedef float f32.
{
    f32 x;
    f32 y;
};

int main()
{
    struct vec2f v = { 1, 2}; //x = 1, y = 2
    printf("Vector = < %f, %f\n", v.x, v.y); //first %f corresponds to first v.x, second to v.y, stupid
// %f floats, %d, integers, %u unsigned int

    return 0;
}
