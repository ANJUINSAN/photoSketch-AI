import React from 'react'
import './breadcrums.css'
import arrow_icon from '../images/Assets/breadcrum_arrow.png'

export default function Breadcrums(props) {
    const {product} =props;
  return (
    <div className='breadcrums'>
      HOME <img src={arrow_icon}></img>
      SHOP <img src={arrow_icon} alt="" />
      {product.category} <img src={arrow_icon} alt="" />
      {product.name}
    </div>
  )
}
