//-------------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//    File      : Matrix.h
//    Project   : FruitBall Android
//    Editor    : Ali Salmanizadegan
//    Date      : 1397\02\06
//    Edit		: 1397\02\06
//    Time      : 10:32:00
//    Copyright : (c) Codeart3D Corporation. All rights reserved.
//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------------

#pragma once
#include <cstring>
#include "Types.h"

// matrix 2x2
void Matrix2Rotation(MATRIX2 *m, float t);
void Vector2MultiplyMatrix2(VECTOR2 *v, MATRIX2 *m);

// matrix 3x3
void Matrix3Zero(MATRIX3 *m);
void Matrix3Identity(MATRIX3 *m);
void Matrix3Quaternion(MATRIX3 *out, VECTOR4 q);
void Vector3MultiplyMatrix3(VECTOR3 *v, MATRIX3 *m);
void Vector3MultiplyMatrix3(VECTOR3 *out, VECTOR3 *v, MATRIX3 *m);
void Vector3MultiplyMatrix4(VECTOR3 *out, VECTOR3 v, MATRIX4 *m);
void Vector3MultiplyMatrix4(VECTOR3 *out, VECTOR3 *v, MATRIX4 *m);
void Vector3TransformCoord(VECTOR3 *pout, const VECTOR3 *pv, const MATRIX4 *pm);
void Vector3TransformNormal(VECTOR3 *pout, const VECTOR3 *pv, const MATRIX4 *pm);

// matrix 4x4
void Matrix4Identity(MATRIX4 *m);
void Matrix4Transpose(MATRIX4 *m);
void Matrix4RotationX(MATRIX4 *out, float x);
void Matrix4RotateFast(MATRIX4 *m, VECTOR4 *v);
void Matrix4Quaternion(MATRIX4 *out, VECTOR4 q);
void Matrix4CopyMatrix3(MATRIX4 *out, MATRIX3 *m);
void Matrix4Scale(MATRIX4 *out, MATRIX4 *m, VECTOR3 *v);
void Matrix4Rotate(MATRIX4 *out, MATRIX4 *m, VECTOR4 *v);
void Matrix4Translate(MATRIX4 *out, MATRIX4 *m, VECTOR3 *v);
void Vector4MultiplyMatrix4(VECTOR4 *out, VECTOR4 v, MATRIX4 *m);
void Vector4MultiplyMatrix4(VECTOR4 *out, VECTOR4 *v, MATRIX4 *m);
void Matrix4MultiplyMatrix3(MATRIX4 *out, MATRIX4 *m1, MATRIX3 *m2);
void Matrix4MultiplyMatrix4(MATRIX4 *out, MATRIX4 *m1, MATRIX4 *m2);
void Matrix3MultiplyMatrix4Fast(MATRIX4 * out, MATRIX4 * m4, MATRIX3 * m3);
void Matrix4MultiplyMatrix4Fast(MATRIX4 * out, MATRIX4 * m1, MATRIX4 * m2);
void Matrix4MultiplyMatrix4Mode2D(MATRIX4 * out, MATRIX4 * m1, MATRIX4 * m2, float& lookdis);
void Matrix4Zoom(MATRIX4 *out, float zoom, float x, float y);
void Matrix4Ortho(MATRIX4 *out, float left, float right, float bottom, float top, float clip_start, float clip_end);

bool Matrix4Invert(MATRIX4 *m);
bool Matrix4InvertFull(MATRIX4 *m);
bool Matrix4Invert(MATRIX4 & inv, MATRIX4 & world);

MATRIX3 Matrix3Translation(VECTOR2 &trans);
MATRIX3 Matrix3TranslationInverse(VECTOR2 &trans);
MATRIX3 Matrix3Rotation(float t);
MATRIX4 Matrix4RotationX(float t);
MATRIX4 Matrix4RotationY(float t);
MATRIX4 Matrix4RotationZ(float t);
MATRIX4 Matrix4Rotation(float Yaw, float Pitch, float Roll);