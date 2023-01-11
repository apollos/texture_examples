/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __CPU_ANIM2_H__
#define __CPU_ANIM2_H__

#include <iostream>


struct CPUAnimBitmap {
    unsigned char    *pixels;
    int     width, height;
    void    *dataBlock;
    void (*fAnim)(void*,int);
    void (*animExit)(void*);
    void (*clickDrag)(void*,int,int,int,int);
    int     dragStartX, dragStartY;

    CPUAnimBitmap( int w, int h, void *d = NULL ) {
        width = w;
        height = h;
        pixels = new unsigned char[width * height * 4];
        dataBlock = d;
        clickDrag = NULL;
    }

    ~CPUAnimBitmap() {
        delete [] pixels;
    }

    unsigned char* get_ptr( void ) const   { return pixels; }
    long image_size( void ) const { return width * height * 4; }

    void click_drag( void (*f)(void*,int,int,int,int)) {
        clickDrag = f;
    }

    void anim_and_exit( void (*f)(void*,int), void(*e)(void*) ) {
        
    }

    // static method used for glut callbacks
    static CPUAnimBitmap** get_bitmap_ptr( void ) {
        static CPUAnimBitmap*   gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void mouse_func( int button, int state,
                            int mx, int my ) {
        
    }

    // static method used for glut callbacks
    static void idle_func( void ) {

    }

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y) {

    }

    // static method used for glut callbacks
    static void Draw( void ) {

    }
};


#endif  // __CPU_ANIM_H__

