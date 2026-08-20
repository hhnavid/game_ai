//-------------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//    File      : Type.cpp
//    Project   : Fruit Ball
//    Author    : Ali Salmanizadegan
//    Date      : 1394\--\--
//    Time      : --:--:--
//    Copyright : (c) Codeart3D Corporation. All rights reserved.
//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------------

#include "stdafx.h"
#include "Types.h"

//
/// VECTOR2 operators
//

VECTOR2 & VECTOR2::operator+=(const VECTOR2 &v)
{
	x += v.x;
	y += v.y;

	return *this;
}

VECTOR2 & VECTOR2::operator-=(const VECTOR2 &v)
{
	x -= v.x;
	y -= v.y;

	return *this;
}

VECTOR2 & VECTOR2::operator*=(float m)
{
	x *= m;
	y *= m;

	return *this;
}

VECTOR2 & VECTOR2::operator/=(float d)
{
	x /= d;
	y /= d;

	return *this;
}

VECTOR2 VECTOR2::operator+() const
{
	return *this;
}

VECTOR2 VECTOR2::operator-() const
{
	return VECTOR2(-x, -y);
}

VECTOR2 VECTOR2::operator+(const VECTOR2 &v) const
{
	return VECTOR2(x + v.x, y + v.y);
}

VECTOR2 VECTOR2::operator-(const VECTOR2 &v) const
{
	return VECTOR2(x - v.x, y - v.y);
}

VECTOR2 VECTOR2::operator*(const VECTOR2 &v) const
{
	return VECTOR2(x * v.x, y * v.y);
}

VECTOR2 VECTOR2::operator*(float m) const
{
	return VECTOR2(x * m, y * m);
}

VECTOR2 VECTOR2::operator/(float d) const
{
	return VECTOR2(x / d, y / d);
}

VECTOR2 operator*(float m, const VECTOR2 &v)
{
	return VECTOR2(v.x * m, v.y * m);
}

bool VECTOR2::operator==(const VECTOR2 &v) const
{
	return (x == v.x && y == v.y);
}

bool VECTOR2::operator!=(const VECTOR2 &v) const
{
	return (x != v.x && y != v.y);
}

//
/// VECTOR3 operators
//

VECTOR3 & VECTOR3::operator+=(const VECTOR3 &v)
{
	x += v.x;
	y += v.y;
	z += v.z;

	return *this;
}

VECTOR3 & VECTOR3::operator-=(const VECTOR3 &v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;

	return *this;
}

VECTOR3 & VECTOR3::operator+=(float m)
{
	x += m;
	y += m;
	z += m;

	return *this;
}

VECTOR3 & VECTOR3::operator-=(float m)
{
	x -= m;
	y -= m;
	z -= m;

	return *this;
}

VECTOR3 & VECTOR3::operator*=(float m)
{
	x *= m;
	y *= m;
	z *= m;

	return *this;
}

VECTOR3 & VECTOR3::operator/=(float d)
{
	x /= d;
	y /= d;
	z /= d;

	return *this;
}

VECTOR3 VECTOR3::operator+() const
{
	return *this;
}

VECTOR3 VECTOR3::operator-() const
{
	return VECTOR3(-x, -y, -z);
}

VECTOR3 VECTOR3::operator+(const VECTOR3 &v) const
{
	return VECTOR3(x + v.x, y + v.y, z + v.z);
}

VECTOR3 VECTOR3::operator-(const VECTOR3 &v) const
{
	return VECTOR3(x - v.x, y - v.y, z - v.z);
}

VECTOR3 VECTOR3::operator*(float m) const
{
	return VECTOR3(x * m, y * m, z * m);
}

VECTOR3 VECTOR3::operator/(float d) const
{
	return VECTOR3(x / d, y / d, z / d);
}

VECTOR3 operator*(float m, const VECTOR3 &v)
{
	return VECTOR3(v.x * m, v.y * m, v.z * m);
}

bool VECTOR3::operator==(const VECTOR3 &v) const
{
	return (x == v.x && y == v.y && z == v.z);
}

bool VECTOR3::operator!=(const VECTOR3 &v) const
{
	return (x != v.x || y != v.y || z != v.z);
}

bool VECTOR3::operator>(const VECTOR3 &v) const
{
	return sqrtf(x * x + y * y + z * z) > sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

bool VECTOR3::operator<(const VECTOR3 &v) const
{
	return sqrtf(x * x + y * y + z * z) < sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

VECTOR3 MATRIX3::operator*(const VECTOR3 &v) const
{
	VECTOR3 out;

	out.x = (v.x * m[0].x) + (v.y * m[1].x) + (v.z * m[2].x);
	out.y = (v.x * m[0].y) + (v.y * m[1].y) + (v.z * m[2].y);
	out.z = (v.x * m[0].z) + (v.y * m[1].z) + (v.z * m[2].z);

	return out;
}

MATRIX3 MATRIX3::operator*(const MATRIX3 &t) const
{
	MATRIX3 mat;

	mat.m[0].x = m[0].x * t.m[0].x + m[1].x * t.m[0].y + m[2].x * t.m[0].z;
	mat.m[0].y = m[0].y * t.m[0].x + m[1].y * t.m[0].y + m[2].y * t.m[0].z;
	mat.m[0].z = m[0].z * t.m[0].x + m[1].z * t.m[0].y + m[2].z * t.m[0].z;

	mat.m[1].x = m[0].x * t.m[1].x + m[1].x * t.m[1].y + m[2].x * t.m[1].z;
	mat.m[1].y = m[0].y * t.m[1].x + m[1].y * t.m[1].y + m[2].y * t.m[1].z;
	mat.m[1].z = m[0].z * t.m[1].x + m[1].z * t.m[1].y + m[2].z * t.m[1].z;

	mat.m[2].x = m[0].x * t.m[2].x + m[1].x * t.m[2].y + m[2].x * t.m[2].z;
	mat.m[2].y = m[0].y * t.m[2].x + m[1].y * t.m[2].y + m[2].y * t.m[2].z;
	mat.m[2].z = m[0].z * t.m[2].x + m[1].z * t.m[2].y + m[2].z * t.m[2].z;

	return mat;
}

MATRIX3 MATRIX3::operator*=(const MATRIX3 &t)
{
	MATRIX3 mat;

	mat.m[0].x = m[0].x * t.m[0].x + m[1].x * t.m[0].y + m[2].x * t.m[0].z;
	mat.m[0].y = m[0].y * t.m[0].x + m[1].y * t.m[0].y + m[2].y * t.m[0].z;
	mat.m[0].z = m[0].z * t.m[0].x + m[1].z * t.m[0].y + m[2].z * t.m[0].z;

	mat.m[1].x = m[0].x * t.m[1].x + m[1].x * t.m[1].y + m[2].x * t.m[1].z;
	mat.m[1].y = m[0].y * t.m[1].x + m[1].y * t.m[1].y + m[2].y * t.m[1].z;
	mat.m[1].z = m[0].z * t.m[1].x + m[1].z * t.m[1].y + m[2].z * t.m[1].z;

	mat.m[2].x = m[0].x * t.m[2].x + m[1].x * t.m[2].y + m[2].x * t.m[2].z;
	mat.m[2].y = m[0].y * t.m[2].x + m[1].y * t.m[2].y + m[2].y * t.m[2].z;
	mat.m[2].z = m[0].z * t.m[2].x + m[1].z * t.m[2].y + m[2].z * t.m[2].z;

	m[0].x = mat.m[0].x;
	m[0].y = mat.m[0].y;
	m[0].z = mat.m[0].z;

	m[1].x = mat.m[1].x;
	m[1].y = mat.m[1].y;
	m[1].z = mat.m[1].z;

	m[2].x = mat.m[2].x;
	m[2].y = mat.m[2].y;
	m[2].z = mat.m[2].z;

	return *this;
}

//
/// VECTOR4 operators
//

VECTOR4 & VECTOR4::operator+=(const VECTOR4 &v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	w += v.w;

	return *this;
}

VECTOR4 & VECTOR4::operator-=(const VECTOR4 &v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	w -= v.w;

	return *this;
}

VECTOR4 & VECTOR4::operator*=(float m)
{
	x *= m;
	y *= m;
	z *= m;
	w *= m;

	return *this;
}

VECTOR4 & VECTOR4::operator/=(float d)
{
	x /= d;
	y /= d;
	z /= d;
	w /= d;

	return *this;
}

VECTOR4 VECTOR4::operator+() const
{
	return *this;
}

VECTOR4 VECTOR4::operator-() const
{
	return VECTOR4(-x, -y, -z, -w);
}

VECTOR4 VECTOR4::operator+(const VECTOR4 &v) const
{
	return VECTOR4(x + v.x, y + v.y, z + v.z, w + v.w);
}

VECTOR4 VECTOR4::operator-(const VECTOR4 &v) const
{
	return VECTOR4(x - v.x, y - v.y, z - v.z, w - v.w);
}

VECTOR4 VECTOR4::operator*(float m) const
{
	return VECTOR4(x * m, y * m, z * m, w * m);
}

VECTOR4 VECTOR4::operator/(float d) const
{
	return VECTOR4(x / d, y / d, z / d, w / d);
}

VECTOR4 operator*(float m, const VECTOR4 &v)
{
	return VECTOR4(v.x * m, v.y * m, v.z * m, v.w * m);
}

VECTOR3 MATRIX4::operator*(const VECTOR3 &v) const
{
	VECTOR3 out;

	out.x = (v.x * m[0].x) + (v.y * m[1].x) + (v.z * m[2].x);
	out.y = (v.x * m[0].y) + (v.y * m[1].y) + (v.z * m[2].y);
	out.z = (v.x * m[0].z) + (v.y * m[1].z) + (v.z * m[2].z);

	return out;
}

VECTOR4 MATRIX4::operator*(const VECTOR4 &v) const
{
	VECTOR4 out;

	out.x = (v.x * m[0].x) + (v.y * m[1].x) + (v.z * m[2].x) + (v.w * m[3].x);
	out.y = (v.x * m[0].y) + (v.y * m[1].y) + (v.z * m[2].y) + (v.w * m[3].y);
	out.z = (v.x * m[0].z) + (v.y * m[1].z) + (v.z * m[2].z) + (v.w * m[3].z);
	out.w = (v.x * m[0].w) + (v.y * m[1].w) + (v.z * m[2].w) + (v.w * m[3].w);

	return out;
}

MATRIX4 MATRIX4::operator*(const MATRIX4 &t) const
{
	MATRIX4 mat;

	mat.m[0].x = m[0].x * t.m[0].x + m[1].x * t.m[0].y + m[2].x * t.m[0].z + m[3].x * t.m[0].w;
	mat.m[0].y = m[0].y * t.m[0].x + m[1].y * t.m[0].y + m[2].y * t.m[0].z + m[3].y * t.m[0].w;
	mat.m[0].z = m[0].z * t.m[0].x + m[1].z * t.m[0].y + m[2].z * t.m[0].z + m[3].z * t.m[0].w;
	mat.m[0].w = m[0].w * t.m[0].x + m[1].w * t.m[0].y + m[2].w * t.m[0].z + m[3].w * t.m[0].w;

	mat.m[1].x = m[0].x * t.m[1].x + m[1].x * t.m[1].y + m[2].x * t.m[1].z + m[3].x * t.m[1].w;
	mat.m[1].y = m[0].y * t.m[1].x + m[1].y * t.m[1].y + m[2].y * t.m[1].z + m[3].y * t.m[1].w;
	mat.m[1].z = m[0].z * t.m[1].x + m[1].z * t.m[1].y + m[2].z * t.m[1].z + m[3].z * t.m[1].w;
	mat.m[1].w = m[0].w * t.m[1].x + m[1].w * t.m[1].y + m[2].w * t.m[1].z + m[3].w * t.m[1].w;

	mat.m[2].x = m[0].x * t.m[2].x + m[1].x * t.m[2].y + m[2].x * t.m[2].z + m[3].x * t.m[2].w;
	mat.m[2].y = m[0].y * t.m[2].x + m[1].y * t.m[2].y + m[2].y * t.m[2].z + m[3].y * t.m[2].w;
	mat.m[2].z = m[0].z * t.m[2].x + m[1].z * t.m[2].y + m[2].z * t.m[2].z + m[3].z * t.m[2].w;
	mat.m[2].w = m[0].w * t.m[2].x + m[1].w * t.m[2].y + m[2].w * t.m[2].z + m[3].w * t.m[2].w;

	mat.m[3].x = m[0].x * t.m[3].x + m[1].x * t.m[3].y + m[2].x * t.m[3].z + m[3].x * t.m[3].w;
	mat.m[3].y = m[0].y * t.m[3].x + m[1].y * t.m[3].y + m[2].y * t.m[3].z + m[3].y * t.m[3].w;
	mat.m[3].z = m[0].z * t.m[3].x + m[1].z * t.m[3].y + m[2].z * t.m[3].z + m[3].z * t.m[3].w;
	mat.m[3].w = m[0].w * t.m[3].x + m[1].w * t.m[3].y + m[2].w * t.m[3].z + m[3].w * t.m[3].w;

	return mat;
}