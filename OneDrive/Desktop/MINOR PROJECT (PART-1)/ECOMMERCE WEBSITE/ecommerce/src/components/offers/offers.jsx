import React from 'react'
import './offers.css'
import exclusive_image from '../images/Assets/exclusive_image.png'
import { Link } from 'react-router-dom'
export default function Offers() {
  return (
    <div>
      <div className='offers'>
        <div className='offers-left'>
            <h1>Exclusive</h1>
            <h1>Offers for you</h1>
            <p>ONLY ON BEST SELLERS PRODUCTS</p>
            <Link  to={'/womens'} style={{textDecoration:'none'}}><button>check now</button></Link>
        </div>
        <div className='offers-right'>
            <img src= {exclusive_image}alt=""  />
        </div>
      </div>
    </div>
  )
}
