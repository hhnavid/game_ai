//-------------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//    File      : Type.hpp
//    Project   : Fruit Ball
//    Author    : Ali Salmanizadegan
//    Date      : 1394\--\--
//	  Edit		: 1397\02\06
//    Time      : --:--:--
//    Copyright : (c) Codeart3D Corporation. All rights reserved.
//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------------

#pragma once

#include <math.h>
#define M_PI 3.14159265358979323846

#define DEG_TO_RAD M_PI / 180.0f
#define RAD_TO_DEG 90.0f / M_PI
#define ATA_TO_DEG 180.0f / M_PI

#ifdef ARM
typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned long DWORD;
typedef unsigned int UINT;
#endif

typedef signed char  int8;
typedef signed short int16;
typedef signed int   int32;
typedef signed long  int64;

// Color Type

struct COLOR
{
	float r;
	float g;
	float b;
	float a;

	COLOR()
	{
		r = 1.0f;
		g = 1.0f;
		b = 1.0f;
		a = 1.0f;
	}

	COLOR(float _r, float _g, float _b, float _a)
	{
		r = _r;
		g = _g;
		b = _b;
		a = _a;
	}

	COLOR(int color)
	{
		r = (float)(BYTE(color >> 24) / 255.0f);
		g = (float)(BYTE(color >> 16) / 255.0f);
		b = (float)(BYTE(color >> 8) / 255.0f);
		a = (float)(BYTE(color) / 255.0f);
	}

	operator int()
	{
		BYTE br = BYTE(r * 255);
		BYTE bg = BYTE(g * 255);
		BYTE bb = BYTE(b * 255);
		BYTE ba = BYTE(a * 255);
		return (int)(br << 24 | bg << 16 | bb << 8 | ba);
	}
};

struct RGBA
{
	BYTE r;
	BYTE g;
	BYTE b;
	BYTE a;

	RGBA()
	{
		r = 255;
		g = 255;
		b = 255;
		a = 255;
	}

	RGBA(const COLOR& c)
	{
		r = BYTE(c.r * 255);
		g = BYTE(c.g * 255);
		b = BYTE(c.b * 255);
		a = BYTE(c.a * 255);
	}

	RGBA(BYTE _r, BYTE _g, BYTE _b, BYTE _a)
	{
		r = _r;
		g = _g;
		b = _b;
		a = _a;
	}

	RGBA(int color)
	{
		r = BYTE(color >> 24);
		g = BYTE(color >> 16);
		b = BYTE(color >> 8);
		a = BYTE(color);
	}

	operator COLOR()
	{
		return COLOR(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
	}

	operator int()
	{
		return (int)(r << 24 | g << 16 | b << 8 | a);
	}
};

struct VECTOR2
{
	float x;
	float y;

	VECTOR2()
	{
		x = 0.0f;
		y = 0.0f;
	}

	VECTOR2(float _x, float _y)
	{
		x = _x;
		y = _y;
	}

	void Identity()
	{
		x = 0.0f;
		y = 0.0f;
	}

	/**
	 * @brief Normalizes the vector so that it will be of unit length
	 */
	void Normalize()
	{
		float norm = sqrtf(x * x + y * y);
		x /= norm;
		y /= norm;
	}

	VECTOR2& operator += (const VECTOR2&);
	VECTOR2& operator -= (const VECTOR2&);
	VECTOR2& operator *= (float);
	VECTOR2& operator /= (float);

	VECTOR2 operator + () const;
	VECTOR2 operator - () const;

	VECTOR2 operator + (const VECTOR2&) const;
	VECTOR2 operator - (const VECTOR2&) const;
	VECTOR2 operator * (const VECTOR2&) const;
	VECTOR2 operator * (float) const;
	VECTOR2 operator / (float) const;

	friend VECTOR2 operator * (float, const VECTOR2&);

	bool operator == (const VECTOR2&) const;
	bool operator != (const VECTOR2&) const;	
};

struct VECTOR3
{
	float x;
	float y;
	float z;

	VECTOR3()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}

	VECTOR3(const float v)
	{
		x = v;
		y = v;
		z = v;
	}

	VECTOR3(const VECTOR2& v)
	{
		x = v.x;
		y = v.y;
		z = 0;
	}
	
	VECTOR3(float _x, float _y, float _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

	void Identity()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}

	void IdentityOne()
	{
		x = 1.0f;
		y = 1.0f;
		z = 1.0f;
	}

	void IdentityUp()
	{
		x = 0.0f;
		y = 0.0f;
		z = 1.0f;
	}

	VECTOR3& operator += (const VECTOR3&);
	VECTOR3& operator -= (const VECTOR3&);
	VECTOR3& operator += (float);
	VECTOR3& operator -= (float);
	VECTOR3& operator *= (float);
	VECTOR3& operator /= (float);

	VECTOR3 operator + () const;
	VECTOR3 operator - () const;

	VECTOR3 operator + (const VECTOR3&) const;
	VECTOR3 operator - (const VECTOR3&) const;
	VECTOR3 operator * (float) const;
	VECTOR3 operator / (float) const;

	friend VECTOR3 operator * (float, const VECTOR3&);

	bool operator == (const VECTOR3&) const;
	bool operator != (const VECTOR3&) const;
	bool operator > (const VECTOR3&) const;
	bool operator < (const VECTOR3&) const;

	operator VECTOR2()
	{
		return VECTOR2(x, y);
	}	

	VECTOR3& operator = (const VECTOR2& v)
	{
		x = v.x;
		y = v.y;
		z = 0;

		return *this;
	}	

	float Length()
	{
		return sqrtf((x * x) + (y * y) + (z * z));
	}

	float Normalize()
	{
		float len = sqrtf((x * x) + (y * y) + (z * z));

		if (len)
		{
			float m = 1.0f / len;
			x *= m;
			y *= m;
			z *= m;
		}

		return len;
	}

	static VECTOR3 Mix(VECTOR3 &u, VECTOR3 &v, float &a)
	{
		return u + a * (v - u);
	}
};

struct VECTOR4
{
	float x;
	float y;
	float z;
	float w;

	VECTOR4()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
		w = 0.0f;
	}

	VECTOR4(float _x, float _y, float _z, float _w)
	{
		x = _x;
		y = _y;
		z = _z;
		w = _w;
	}
	VECTOR4(const COLOR& c)
	{
		x = c.r;
		y = c.g;
		z = c.b;
		w = c.a;
	}
	VECTOR4(const RGBA& c)
	{
		x = c.r;
		y = c.g;
		z = c.b;
		w = c.a;
	}
	VECTOR4(const VECTOR2& v)
	{
		x = v.x;
		y = v.y;
		z = 0.0f;
		w = 0.0f;
	}
	VECTOR4(const VECTOR2& v, float _z, float _w)
	{
		x = v.x;
		y = v.y;
		z = _z;
		w = _w;
	}
	VECTOR4(const VECTOR3& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
		w = 0.0f;
	}
	VECTOR4(const VECTOR3& v, float _w)
	{
		x = v.x;
		y = v.y;
		z = v.z;
		w = _w;
	}

	void Identity()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
		w = 0.0f;
	}

	VECTOR4& operator += (const VECTOR4&);
	VECTOR4& operator -= (const VECTOR4&);
	VECTOR4& operator *= (float);
	VECTOR4& operator /= (float);

	VECTOR4 operator + () const;
	VECTOR4 operator - () const;

	VECTOR4 operator + (const VECTOR4&) const;
	VECTOR4 operator - (const VECTOR4&) const;
	VECTOR4 operator * (float) const;
	VECTOR4 operator / (float) const;

	friend VECTOR4 operator * (float, const VECTOR4&);

	operator COLOR()
	{
		return COLOR(x, y, z, w);
	}
	operator VECTOR2()
	{
		return VECTOR2(x, y);
	}
	operator VECTOR3()
	{
		return VECTOR3(x, y, z);
	}

	VECTOR4& operator = (const VECTOR2& v)
	{
		x = v.x;
		y = v.y;

		return *this;
	}

	VECTOR4& operator = (const VECTOR3& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;

		return *this;
	}

	float Length()
	{
		return sqrtf((x * x) + (y * y) + (z * z) + (w * w));
	}

	float Normalize()
	{
		float len = sqrtf((x * x) + (y * y) + (z * z) + (w * w));
		float m = len ? 1.0f / len : 0.0f;
		x *= m;
		y *= m;
		z *= m;
		w *= m;

		return len;
	}
};

struct MATRIX2
{
	VECTOR2 m[2];

	MATRIX2() {};
	MATRIX2(float m00, float m01, float m10, float m11)
	{
		m[0].x = m00;
		m[0].y = m01;

		m[1].x = m10;
		m[1].y = m11;
	};
};

struct MATRIX3
{
	VECTOR3 m[3];

	MATRIX3() {};
	MATRIX3(float m00, float m01, float m02,
		float m10, float m11, float m12,
		float m20, float m21, float m22)
	{
		m[0].x = m00;
		m[0].y = m01;
		m[0].z = m02;

		m[1].x = m10;
		m[1].y = m11;
		m[1].z = m12;

		m[2].x = m20;
		m[2].y = m21;
		m[2].z = m22;
	};

	MATRIX3(VECTOR3 m1, VECTOR3 m2, VECTOR3 m3)
	{
		m[0] = m1;
		m[1] = m2;
		m[2] = m3;
	};

	VECTOR3 operator * (const VECTOR3&) const;
	MATRIX3 operator * (const MATRIX3&) const;
	MATRIX3 operator *= (const MATRIX3&);
};

struct MATRIX4
{
	VECTOR4 m[4];

	MATRIX4() {};
	MATRIX4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33)
	{
		m[0].x = m00;
		m[0].y = m01;
		m[0].z = m02;
		m[0].w = m03;

		m[1].x = m10;
		m[1].y = m11;
		m[1].z = m12;
		m[1].w = m13;

		m[2].x = m20;
		m[2].y = m21;
		m[2].z = m22;
		m[2].w = m23;

		m[3].x = m30;
		m[3].y = m31;
		m[3].z = m32;
		m[3].w = m33;
	};

	void Identity()
	{
		m[0].y = 0.0f;
		m[0].z = 0.0f;
		m[0].w = 0.0f;

		m[1].x = 0.0f;
		m[1].z = 0.0f;
		m[1].w = 0.0f;

		m[2].x = 0.0f;
		m[2].y = 0.0f;
		m[2].w = 0.0f;

		m[3].x = 0.0f;
		m[3].y = 0.0f;
		m[3].z = 0.0f;

		m[0].x = 1.0f;
		m[1].y = 1.0f;
		m[2].z = 1.0f;
		m[3].w = 1.0f;
	}

	void Identity(VECTOR3& move)
	{
		m[0].y = 0.0f;
		m[0].z = 0.0f;
		m[0].w = 0.0f;

		m[1].x = 0.0f;
		m[1].z = 0.0f;
		m[1].w = 0.0f;

		m[2].x = 0.0f;
		m[2].y = 0.0f;
		m[2].w = 0.0f;

		m[3].x = move.x;
		m[3].y = move.y;
		m[3].z = move.z;

		m[0].x = 1.0f;
		m[1].y = 1.0f;
		m[2].z = 1.0f;
		m[3].w = 1.0f;
	}

	static MATRIX4 GetIdentity()
	{
		MATRIX4 m;
		m.Identity();

		return m;
	}

	MATRIX4 Translate(VECTOR3 &v)
	{
		MATRIX4 out = *this;

		out.m[3].x = m[0].x * v.x + m[1].x * v.y + m[2].x * v.z + m[3].x;
		out.m[3].y = m[0].y * v.x + m[1].y * v.y + m[2].y * v.z + m[3].y;
		out.m[3].z = m[0].z * v.x + m[1].z * v.y + m[2].z * v.z + m[3].z;
		out.m[3].w = m[0].w * v.x + m[1].w * v.y + m[2].w * v.z + m[3].w;

		return out;
	}

	MATRIX4 Rotate(VECTOR4 &v)
	{
		float value = v.w * DEG_TO_RAD;
		float s = sinf(value);
		float c = cosf(value);
		float xx, yy, zz, xy, yz, zx, xs, ys, zs, c1;

		MATRIX4 mat;
		mat.Identity();
		VECTOR3 t = VECTOR3(v.x, v.y, v.z);

		if (!v.w || !t.Normalize())
			return *this;

		xx = t.x * t.x;
		yy = t.y * t.y;
		zz = t.z * t.z;
		xy = t.x * t.y;
		yz = t.y * t.z;
		zx = t.z * t.x;
		xs = t.x * s;
		ys = t.y * s;
		zs = t.z * s;
		c1 = 1.0f - c;

		mat.m[0].x = (c1 * xx) + c;
		mat.m[1].x = (c1 * xy) - zs;
		mat.m[2].x = (c1 * zx) + ys;

		mat.m[0].y = (c1 * xy) + zs;
		mat.m[1].y = (c1 * yy) + c;
		mat.m[2].y = (c1 * yz) - xs;

		mat.m[0].z = (c1 * zx) - ys;
		mat.m[1].z = (c1 * yz) + xs;
		mat.m[2].z = (c1 * zz) + c;

		return (*this) * mat;
	}

	MATRIX4 Scale(VECTOR3 &v)
	{
		MATRIX4 out;

		out.m[0].x = m[0].x * v.x;
		out.m[0].y = m[0].y * v.x;
		out.m[0].z = m[0].z * v.x;
		out.m[0].w = m[0].w * v.x;

		out.m[1].x = m[1].x * v.y;
		out.m[1].y = m[1].y * v.y;
		out.m[1].z = m[1].z * v.y;
		out.m[1].w = m[1].w * v.y;

		out.m[2].x = m[2].x * v.z;
		out.m[2].y = m[2].y * v.z;
		out.m[2].z = m[2].z * v.z;
		out.m[2].w = m[2].w * v.z;

		out.m[3] = m[3];

		return out;
	}

	MATRIX4 Quaternion(VECTOR4 q)
	{
		MATRIX4 out = *this;

		float qxx(q.x * q.x);
		float qyy(q.y * q.y);
		float qzz(q.z * q.z);
		float qxz(q.x * q.z);
		float qxy(q.x * q.y);
		float qyz(q.y * q.z);
		float qwx(q.w * q.x);
		float qwy(q.w * q.y);
		float qwz(q.w * q.z);

		out.m[0].x = 1.0f - 2.0f * (qyy + qzz);
		out.m[0].y = 2.0f * (qxy + qwz);
		out.m[0].z = 2.0f * (qxz - qwy);

		out.m[1].x = 2.0f * (qxy - qwz);
		out.m[1].y = 1.0f - 2.0f * (qxx + qzz);
		out.m[1].z = 2.0f * (qyz + qwx);

		out.m[2].x = 2.0f * (qxz + qwy);
		out.m[2].y = 2.0f * (qyz - qwx);
		out.m[2].z = 1.0f - 2.0f * (qxx + qyy);

		return out;
	}

	operator MATRIX3()
	{
		return MATRIX3(m[0], m[1], m[2]);
	}

	MATRIX4 operator * (const MATRIX4&) const;

	VECTOR3 operator * (const VECTOR3&) const;
	VECTOR4 operator * (const VECTOR4&) const;
};

struct BoundingBox
{
	VECTOR3 min;
	VECTOR3 max;
};

struct PLANE
{
	// Normal vector
	VECTOR3 n;
	// One point on the plane
	VECTOR3 p;

	PLANE() {}
	PLANE(VECTOR3 normal, VECTOR3 point)
	{
		n = normal;
		p = point;
	}
};